/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <new>
#include <utility>
#include "ub_conn_lite.h"
#if defined(CCL_KERNEL_AICPU) && HCOMM_ENABLE_AICPU_PARALLEL_SQ_COPY
#include "aicpu_sharder.h"
#include "aicpu_schedule/aicpu_context.h"
#endif
#include "log.h"
#include "exception_util.h"
#include "udma_data_struct.h"
#include "internal_exception.h"
#include "string_util.h"
#include "binary_stream.h"
#include "data_type.h"
#include "sal.h"

constexpr u32 MAX_LOG_TIMEOUT_MS        = 500;
namespace Hccl {
constexpr u32 ADDR_BIT_OFFSET            = 32;
constexpr u32 SQE_SIZE_128               = 128;
constexpr u32 SQE_SIZE_64                = 64;
constexpr u32 SQE_INLINE_DATA_SIZE       = 16;
constexpr u32 RAW_SIZE                   = 16;
constexpr u32 RMT_EID_BYTE_SIZE          = 16;
constexpr u32 PI_NUM_TWO                 = 2;
constexpr u32 WRITE_WITH_NOTIFY_OPCODE   = 0x5;
constexpr u32 ADDR_BIT_LOW               = 0xffffffff;
constexpr u32 UB_DMA_MAX_READ_WEITE_SIZE = 256 * 1024 * 1024; // Byte, UB协议一次传输的最大size
constexpr u32 UB_RELAX_ORDER             = 0x1; // Relax Order表示当前SQE与后续Strong Order SQE有保序要求
constexpr u32 UB_STRONG_ORDER            = 0x2; // Strong Order表示当前SQE有保序要求，该SQE不能超越前面的Relax Order SQE

static_assert(sizeof(UdmaSqeWrite) == SQE_SIZE_64, "UB READ WQE must occupy one 64-byte SQ slot");

namespace {
struct BatchSqCopySpan {
    u32 stagingIndex{0};
    u32 sqOffset{0};
    u32 wqeCount{0};
};

struct BatchSqCopyJob {
    BatchSqCopySpan spans[2]{};
    u32 spanCount{0};
};

struct alignas(64) BatchSqCopyJobResult {
    bool entered{false};
    bool callbackEntered{false};
    u64 bytes{0};
    u64 calls{0};
    s32 ret{0};
    bool completed{false};
};
static_assert(sizeof(BatchSqCopyJobResult) % 64 == 0,
    "batch SQ copy job results must occupy whole cache lines");

bool IsLegalBatchSqCopyParallelism(u32 parallelism)
{
    return parallelism == 1 || parallelism == 2 || parallelism == 4 || parallelism == 8 || parallelism == 16;
}

s32 GetCurrentAicpuIndex()
{
#if defined(CCL_KERNEL_AICPU) && HCOMM_ENABLE_AICPU_PARALLEL_SQ_COPY
    const u32 threadIndex = aicpu::GetAicpuThreadIndex();
    return threadIndex <= static_cast<u32>(std::numeric_limits<s32>::max())
        ? static_cast<s32>(threadIndex) : -1;
#else
    return -1;
#endif
}
} // namespace

static std::map<DataType, u32> g_ubmaDataTypeMap
    = {{DataType::INT8, 0x0},   {DataType::INT16, 0x1},   {DataType::INT32, 0x2}, {DataType::UINT8, 0x3},
       {DataType::UINT16, 0x4}, {DataType::UINT32, 0x5},  {DataType::FP16, 0x6},  {DataType::FP32, 0x7},
       {DataType::BFP16, 0x8},  {DataType::BF16_SAT, 0x9}};

static std::map<ReduceOp, u32> g_ubmaDataOpMap = {{ReduceOp::SUM, 0xA}, {ReduceOp::MAX, 0x8}, {ReduceOp::MIN, 0x9}};

bool UbConnLite::IsLegalWqeStagingChunk(u32 chunk)
{
    return chunk == 0 || chunk == 1 || chunk == 2 || chunk == 8 || chunk == 16 || chunk == 32;
}

void UbConnLite::FillCommSqe(UdmaSqeCommon *sqe, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg, u32 opCode,
                             SlicePosition slicePos)
{
    u32 cqeEn = (cfg.cqeEn && (slicePos == SlicePosition::LAST || slicePos == SlicePosition::ONLY)) ? 1 : 0;
    sqe->cqe       = cqeEn;
    sqe->owner     = (pi == (sqDepth_ - 1)) ? 1 : 0;
    sqe->opcode    = opCode;
    sqe->tpn       = tpn_;

    // 当前片是ONLY片(只有一片的情况)和最后一片的情况，全严格保序
    if (slicePos == SlicePosition::LAST) {
        sqe->placeOdr = UB_STRONG_ORDER;
        sqe->compOrder = 1;
        sqe->fence = 1;
    } else {
        // 中间片写死配置，第一片由全局cfg配置
        sqe->placeOdr = (slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr;
        sqe->compOrder = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder;
        sqe->fence = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence;
    }

    sqe->se           = 1; // 表示是否使能solicited event
    sqe->rmtJettyType = 1; // 00 JFR  01:JETTY  10:jettyGroup 11:reserved
    s32 ret           = memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtEid_.raw, RAW_SIZE);
    if (UNLIKELY(ret != 0)) {
        HCCL_ERROR("UbConnLite::FillCommSqe FillCommSqe memcpy failed, ret=%d", ret);
        THROW<InternalException>(StringFormat("UbConnLite::FillCommSqe memcpy_sp failed, ret = %d", ret));
    }

    sqe->sgeNum        = 1;
    sqe->targetHint    = 0;
    sqe->rmtObjId      = rmt.GetTokenId();
    sqe->tokenEn       = 1;
    sqe->rmtTokenValue = rmt.GetTokenValue();
    sqe->rmtAddrLow    = rmt.GetAddr() & ADDR_BIT_LOW;
    sqe->rmtAddrHigh   = rmt.GetAddr() >> ADDR_BIT_OFFSET;
    HCCL_INFO("UbConnLite FillCommSqe UdmaSqeCommon slicePos[%d] sqe->cqe = %u, sqe->owner = %u sqe->opcode = %u, "
              "sqe->tpn = %u, sqe->rmtObjId = %u, sqe->rmtAddrLow = %u, sqe->rmtAddrHigh = %u, sqe->placeOdr = %u, "
              "sqe->compOrder = %u, sqe->fence = %u", slicePos, sqe->cqe, sqe->owner, sqe->opcode, sqe->tpn,
              sqe->rmtObjId, sqe->rmtAddrLow, sqe->rmtAddrHigh, sqe->placeOdr, sqe->compOrder, sqe->fence);
}

void UbConnLite::FillCommSqeReduceInfo(UdmaSqeCommon &sqeComm, ReduceOp reduceOp, DataType dataType, u32 udfType) const
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    sqeComm.inlinedata.udfData.udfType    = udfType; // 0代表inline reduce

    if ((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end()) && (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end())) {
        sqeComm.inlinedata.udfData.reduceOp = g_ubmaDataOpMap.at(reduceOp);
        sqeComm.inlinedata.udfData.reduceType = g_ubmaDataTypeMap.at(dataType);
    } else {
        THROW<InvalidParamsException>(StringFormat("%s reduceOp[%s] or type[%s] is not supported.", __func__, reduceOp.Describe().c_str(), dataType.Describe().c_str()));
    }
    
    // udf字段是否有效
    sqeComm.udfFlag = 1;

    HCCL_INFO("[UbConnLite::%s] end, reduceOp[%s], reduceType[%s]", __func__, reduceOp.Describe().c_str(),
              dataType.Describe().c_str());
}

void UbConnLite::ProcessSlices(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, u32 maxSliceSize,
    std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSlice,
    DataType dataType) const
{
    (void)dataType;
    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u64 sliceSize = static_cast<u64>(maxSliceSize);

    u64 locBufSize    = loc.GetSize();
    u64 sliceNum      = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;

    u64 totalSize = sliceNum * sliceSize;

    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        u64 offset = sliceIdx * sliceSize;
        u64 locAddr = loc.GetAddr() + offset;
        u64 rmtAddr = rmt.GetAddr() + offset;

        HCCL_INFO("[UbConnLite::%s] Slice[%llu]: offset=0x%llx, locAddr=0x%llx, rmtAddr=0x%llx, size=0x%llx",
                    __func__, sliceIdx, offset, locAddr, rmtAddr, sliceSize);

        RmaBufSliceLite locSlice(locAddr, sliceSize, 0, loc.GetTokenId());
        
        RmtRmaBufSliceLite rmtSlice(rmtAddr, sliceSize, 0, rmt.GetTokenId(),
                                    rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos;
        slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        if ((sliceIdx == sliceNum - 1) && lastSliceSize == 0) {
            // SlicePosition::ONLY表示既是首片又是尾片的情况，只有一片的情况
            slicePos = (sliceIdx == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        }
        processOneSlice(locSlice, rmtSlice, slicePos);
    }

    if (lastSliceSize > 0) {
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite lastRmtSlice(rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(),
                                        rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos;
        slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        processOneSlice(lastLocSlice, lastRmtSlice, slicePos);
        sliceNum++;
    }

    HCCL_INFO("[UbConnLite::%s] end, locBufSize[%u], sliceNUm[%u], sliceSize[%u], lastSliceSize[%u]", __func__,
              locBufSize, sliceNum, sliceSize, lastSliceSize);
}

void UbConnLite::ProcessSlicesWithNotify(
    const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, u32 maxSliceSize,
    std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSlice,
    std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSliceWithNotify,
    DataType                                                                 dataType) const
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u32 sliceSize = maxSliceSize;
    if (dataType != DataType::INVALID) {
        u32 dataTypeSize = DATA_TYPE_SIZE_MAP.at(dataType);
        sliceSize        = maxSliceSize / dataTypeSize * dataTypeSize;
    }

    u32 locBufSize    = loc.GetSize();
    u32 sliceNum      = locBufSize / sliceSize;
    u32 lastSliceSize = locBufSize % sliceSize;
    if (sliceNum > 0 && lastSliceSize == 0) {
        sliceNum--;
        lastSliceSize = sliceSize;
    }
    u64 totalSize = static_cast<u64>(sliceNum) * static_cast<u64>(sliceSize);
    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        RmaBufSliceLite locSlice(loc.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite rmtSlice(rmt.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, rmt.GetTokenId(),
                                    rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos;
        slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        processOneSlice(locSlice, rmtSlice, slicePos);
    }

    if (lastSliceSize > 0) {
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite lastRmtSlice(rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(),
                                        rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos;
        slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        processOneSliceWithNotify(lastLocSlice, lastRmtSlice, slicePos);
    }

    HCCL_INFO("[UbConnLite::%s] end, locBufSize[%u], sliceNUm[%u], sliceSize[%u], lastSliceSize[%u]", __func__,
              locBufSize, sliceNum, sliceSize, lastSliceSize);
}

void UbConnLite::FillOneSqeWrite(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                                  UdmaSqeWrite *sqe, UdmaSqOpcode opCode, SlicePosition slicePos)
{
    HCCL_INFO("[UbConnLite::%s] start, loc size[%llu]", __func__, loc.GetSize());

    sqe->comm.inlineEn = 0;
    FillCommSqe(&(sqe->comm), rmt, cfg, opCode, slicePos);
    FillLocalSgeSqe(&(sqe->u.sge), loc);
    if (sqe->u.sge.length == 0) {
        sqe->comm.sgeNum = 0;
    }

    HCCL_INFO("[UbConnLite::%s] end", __func__);
}

void UbConnLite::ProcessOneWqe(UdmaSqeWrite *sqe, UdmaSqOpcode opCode, const StreamLite &stream)
{
    (void)stream;
    HCCL_INFO("[UbConnLite::%s] start, opCode[%s]", __func__, opCode.Describe().c_str());

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    u32 sqOffset = pi % sqDepth_;
    if (sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_) {
        piDetourCount++;
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    pi = pi + 1;

    // 写wqe到va
    u8 *va = reinterpret_cast<u8 *>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (!dwqeCacheLocked_) {
        auto ret = memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64);
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));
        }
    }

    HCCL_INFO("[UbConnLite::%s] end, pi[%u], ci[%u]", __func__, pi, ci);
}

void UbConnLite::ProcessOneWqeWithNotify(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt,
                                         const SqeConfigLite &cfg, UdmaSqeWriteWithNotify *sqe,
                                         const RmtRmaBufSliceLite &notify, u64 notifyData, u32 opCode,
                                         SlicePosition slicePos, const StreamLite &stream)
{
    (void)stream;
    HCCL_INFO("[UbConnLite::%s] start, locSize[%u], opCode[%u]", __func__, loc.GetSize(), opCode);

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    u32 sqOffset = pi % sqDepth_; 
    if (sqOffset < sqDepth_ && (sqOffset + PI_NUM_TWO) >= sqDepth_) {
        piDetourCount++;
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    pi = pi + PI_NUM_TWO; 
    // 填充sqe
    sqe->comm.inlineEn = 0;
    FillCommSqe(&(sqe->comm), rmt, cfg, WRITE_WITH_NOTIFY_OPCODE, slicePos);
    FillNotifySqe(&(sqe->notify), notify, notifyData);
    FillLocalSgeSqe(&(sqe->localU.sge), loc);
    if (sqe->localU.sge.length == 0) {
        sqe->comm.sgeNum = 0;
    }
    sqe->rsv1 = 0;
    sqe->rsv2 = 0;

    u8 *va = reinterpret_cast<u8 *>((sqVa_) + sqOffset * SQE_SIZE_64);
    if (!dwqeCacheLocked_) {
        // 带notify的wqe是96字节, 需要占用两个wqebb, 实际占用128字节
        if (sqOffset == sqDepth_ - 1) {
            MemorySetAndCopy(va, SQE_SIZE_64, sqe);
            va  = reinterpret_cast<u8 *>(sqVa_);
            MemorySetAndCopy(va, SQE_SIZE_64, reinterpret_cast<u8 *>(sqe) + SQE_SIZE_64);
        } else {
            MemorySetAndCopy(va, SQE_SIZE_128, sqe);
        }
    }

    HCCL_INFO("[UbConnLite::%s] end, pi[%u], ci[%u]", __func__, pi, ci);
}

void UbConnLite::MemorySetAndCopy(u8 *va, u32 sqeSize, void *sqe)
{
    auto ret = memset_s(va, sqeSize, 0, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] memset fail, ret = %d", __func__, ret));
    }
    ret = memcpy_sp(va, sqeSize, sqe, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    }
}

void UbConnLite::Read(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                      const StreamLite &stream, ConnLiteOperationOut &out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlices(loc, rmt, maxReadSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
        UdmaSqeWrite sqe{};
        FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
        ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);
    });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::BatchRead(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
    const SqeConfigLite &cfg, u32 lastDescriptorIndex, const StreamLite &stream, ConnLiteOperationOut &out,
    u32 stagingChunk, bool parallelCopy, u32 parallelism, BatchSqCopyBarrier workerCompletionBarrier,
    BatchSqCopyBarrier publisherBarrier, bool schedulerJoinGuaranteed, bool diagnosticMode)
{
    batchStagingStats_ = {};
    batchStagingStats_.stagingChunk = stagingChunk;
    batchStagingStats_.requestedParallelism = parallelism;
    if (loc.size() != rmt.size()) {
        THROW<InternalException>(StringFormat(
            "[UbConnLite::%s] local/remote size mismatch, loc[%llu], rmt[%llu]", __func__, loc.size(), rmt.size()));
    }
    if (loc.empty()) {
        out.pi = pi;
        return;
    }
    if (loc.size() > UINT32_MAX || lastDescriptorIndex >= loc.size()) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] invalid descriptor count[%llu] or last index[%u]",
            __func__, loc.size(), lastDescriptorIndex));
    }

    auto processDirect = [&]() {
        for (u64 i = 0; i < loc.size(); ++i) {
            const bool isLastDescriptor = (i == lastDescriptorIndex);
            SqeConfigLite wqeCfg = cfg;
            wqeCfg.cqeEn = isLastDescriptor;
            wqeCfg.placeOdr = isLastDescriptor ? UB_STRONG_ORDER : UB_RELAX_ORDER;
            wqeCfg.compOrder = isLastDescriptor ? 1 : 0;
            Read(loc[i], rmt[i], wqeCfg, stream, out);
        }
    };

    if (stagingChunk == 0) {
        processDirect();
        return;
    }
    if (UNLIKELY(!IsLegalWqeStagingChunk(stagingChunk) || stagingChunk > WQE_STAGING_CHUNK_MAX)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] invalid staging chunk[%u]",
            __func__, stagingChunk));
    }

    bool allSmallRead = true;
    bool hasWqe = false;
    for (const auto &localSlice : loc) {
        if (localSlice.GetSize() != 0) {
            hasWqe = true;
        }
        if (localSlice.GetSize() > maxReadSize) {
            allSmallRead = false;
            break;
        }
    }
    if (UNLIKELY(maxReadSize == 0 && hasWqe)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] invalid read limit[%u]", __func__, maxReadSize));
    }
    if (!allSmallRead) {
        processDirect();
        return;
    }
    if (!hasWqe) {
        out.pi = pi;
        return;
    }
    if (UNLIKELY(maxReadSize == 0 || sqDepth_ == 0)) {
        THROW<InternalException>(
            StringFormat("[UbConnLite::%s] invalid read limit[%u] or SQ depth[%u]", __func__, maxReadSize, sqDepth_));
    }

    if (parallelCopy) {
        BatchReadParallel(loc, rmt, cfg, lastDescriptorIndex, stream, out, stagingChunk, parallelism,
                          workerCompletionBarrier, publisherBarrier, schedulerJoinGuaranteed, diagnosticMode);
        return;
    }

    batchStagingStats_.stagingUsed = true;
    // One timestamp pair is enough to report the timer probe without adding a timestamp per WQE.
    const u64 timerProbeStartNs = GetCurAicpuTimestamp();
    const u64 timerProbeEndNs = GetCurAicpuTimestamp();
    batchStagingStats_.timerProbeNs = timerProbeEndNs - timerProbeStartNs;

    const u16 batchStartPi = pi;
    const u32 batchStartPiDetourCount = piDetourCount;
    constexpr u32 stagingAlignment = 64;
    constexpr u32 stagingStorageSize = WQE_STAGING_CHUNK_MAX * sizeof(UdmaSqeWrite) + stagingAlignment - 1;
    u8 stagingStorage[stagingStorageSize]{};
    const u64 stagingStorageAddr = reinterpret_cast<u64>(stagingStorage);
    const u64 stagingAddr = (stagingStorageAddr + stagingAlignment - 1)
        & ~static_cast<u64>(stagingAlignment - 1);
    UdmaSqeWrite *staging = reinterpret_cast<UdmaSqeWrite *>(stagingAddr);
    for (u32 i = 0; i < WQE_STAGING_CHUNK_MAX; ++i) {
        ::new (static_cast<void *>(staging + i)) UdmaSqeWrite;
    }
    if (UNLIKELY((stagingAddr & (stagingAlignment - 1)) != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] staging address is not 64-byte aligned, addr[%p]",
            __func__, staging));
    }
    u64 descriptorIndex = 0;
    try {
        while (descriptorIndex < loc.size()) {
            const u32 sqOffset = static_cast<u32>(pi) % sqDepth_;
            const u32 sqRemaining = sqDepth_ - sqOffset;
            const u32 chunkLimit = (sqRemaining < stagingChunk) ? sqRemaining : stagingChunk;
            if (UNLIKELY(chunkLimit == 0)) {
                THROW<InternalException>(StringFormat(
                    "[UbConnLite::%s] invalid staging chunk, sqOffset[%u], sqDepth[%u]", __func__, sqOffset, sqDepth_));
            }

            u32 builtWqeCount = 0;
            const u64 buildStartNs = GetCurAicpuTimestamp();
            while (builtWqeCount < chunkLimit && descriptorIndex < loc.size()) {
                if (loc[descriptorIndex].GetSize() == 0) {
                    ++descriptorIndex;
                    continue;
                }

                staging[builtWqeCount] = {};
                const bool isLastDescriptor = (descriptorIndex == lastDescriptorIndex);
                SqeConfigLite wqeCfg = cfg;
                wqeCfg.cqeEn = isLastDescriptor;
                wqeCfg.placeOdr = isLastDescriptor ? UB_STRONG_ORDER : UB_RELAX_ORDER;
                wqeCfg.compOrder = isLastDescriptor ? 1 : 0;
                FillOneSqeWrite(loc[descriptorIndex], rmt[descriptorIndex], wqeCfg, &staging[builtWqeCount],
                    UdmaSqOpcode::UDMA_OPC_READ, SlicePosition::ONLY);

                // FillCommSqe historically derives owner from logical PI. Staging has not committed
                // the chunk yet, so use the physical ring slot explicitly for every cached WQE.
                staging[builtWqeCount].comm.owner
                    = (static_cast<u64>(sqOffset) + builtWqeCount + 1 == sqDepth_) ? 1 : 0;
                ++builtWqeCount;
                ++descriptorIndex;
            }
            if (builtWqeCount == 0) {
                continue;
            }
            batchStagingStats_.stagingBuildNs += GetCurAicpuTimestamp() - buildStartNs;

            const u64 copySize64 = static_cast<u64>(builtWqeCount) * SQE_SIZE_64;
            const u64 sqByteOffset = static_cast<u64>(sqOffset) * SQE_SIZE_64;
            const u64 maxAddress = std::numeric_limits<u64>::max();
            if (UNLIKELY(copySize64 > UINT32_MAX || sqVa_ > maxAddress - sqByteOffset
                         || copySize64 > maxAddress - (sqVa_ + sqByteOffset))) {
                THROW<InternalException>(StringFormat("[UbConnLite::%s] staging copy arithmetic overflow, pi[%u], "
                                                      "sqOffset[%u], wqeCount[%u], sqDepth[%u]",
                    __func__, pi, sqOffset, builtWqeCount, sqDepth_));
            }

            if (!dwqeCacheLocked_) {
                u8 *va = reinterpret_cast<u8 *>(sqVa_ + sqByteOffset);
                const u32 copyBytes = static_cast<u32>(copySize64);
                const u64 copyStartNs = GetCurAicpuTimestamp();
                const s32 ret = memcpy_sp(va, copyBytes, staging, copyBytes);
                if (UNLIKELY(ret != 0)) {
                    THROW<InternalException>(
                        StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));
                }

                batchStagingStats_.bulkCopyNs += GetCurAicpuTimestamp() - copyStartNs;
                ++batchStagingStats_.bulkCopyCalls;
                batchStagingStats_.bulkCopyWqeCount += builtWqeCount;
            }

            if (static_cast<u64>(sqOffset) + builtWqeCount == sqDepth_) {
                ++piDetourCount;
                ++batchStagingStats_.ringWrapCount;
            }
            // PI is committed only after the complete chunk has been copied successfully.
            pi = static_cast<u16>(static_cast<u32>(pi) + builtWqeCount);
        }
        out.pi = pi;
    } catch (...) {
        // A successful earlier chunk may already occupy SQ slots without a DB task. Restore
        // the logical producer state so a retry starts at the original slot and overwrites it.
        pi = batchStartPi;
        piDetourCount = batchStartPiDetourCount;
        out.pi = batchStartPi;
        throw;
    }
}

void UbConnLite::BatchReadParallel(const vector<RmaBufSliceLite> &loc,
    const vector<RmtRmaBufSliceLite> &rmt, const SqeConfigLite &cfg, u32 lastDescriptorIndex,
    const StreamLite &stream, ConnLiteOperationOut &out, u32 stagingChunk, u32 parallelism,
    BatchSqCopyBarrier workerCompletionBarrier, BatchSqCopyBarrier publisherBarrier,
    bool schedulerJoinGuaranteed, bool diagnosticMode)
{
#if !HCOMM_ENABLE_AICPU_PARALLEL_SQ_COPY
    (void)loc;
    (void)rmt;
    (void)cfg;
    (void)lastDescriptorIndex;
    (void)stream;
    (void)out;
    (void)stagingChunk;
    (void)parallelism;
    (void)workerCompletionBarrier;
    (void)publisherBarrier;
    (void)schedulerJoinGuaranteed;
    (void)diagnosticMode;
    THROW<InternalException>(StringFormat(
        "[UbConnLite::%s] parallel SQ copy compile gate is disabled", __func__));
#elif !defined(CCL_KERNEL_AICPU)
    (void)loc;
    (void)rmt;
    (void)cfg;
    (void)lastDescriptorIndex;
    (void)stream;
    (void)out;
    (void)stagingChunk;
    (void)parallelism;
    (void)workerCompletionBarrier;
    (void)publisherBarrier;
    (void)schedulerJoinGuaranteed;
    (void)diagnosticMode;
    THROW<InternalException>(StringFormat("[UbConnLite::%s] parallel copy is only available in AICPU kernel",
        __func__));
#else
    (void)stream;
    batchStagingStats_.stagingUsed = true;
    batchStagingStats_.parallelCopyUsed = true;
    batchStagingStats_.stagingChunk = stagingChunk;
    batchStagingStats_.requestedParallelism = parallelism;
    batchStagingStats_.diagnosticMode = diagnosticMode;
    const u16 batchStartPi = pi;
    const u32 batchStartPiDetourCount = piDetourCount;
    try {
        if (UNLIKELY(!schedulerJoinGuaranteed)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] parallel SQ copy requires a runtime callback-join contract", __func__));
        }
        if (UNLIKELY(workerCompletionBarrier == nullptr || publisherBarrier == nullptr)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] parallel SQ copy requires worker and publisher barriers", __func__));
        }
        if (UNLIKELY(!IsLegalBatchSqCopyParallelism(parallelism))) {
            THROW<InternalException>(StringFormat("[UbConnLite::%s] invalid parallelism[%u]",
                __func__, parallelism));
        }
        if (UNLIKELY(stagingChunk == 0 || !IsLegalWqeStagingChunk(stagingChunk)
                     || stagingChunk > WQE_STAGING_CHUNK_MAX)) {
            THROW<InternalException>(StringFormat("[UbConnLite::%s] invalid staging chunk[%u]",
                __func__, stagingChunk));
        }
        if (UNLIKELY(dwqeCacheLocked_)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] parallel copy does not support dwqeCacheLocked", __func__));
        }
        constexpr u32 piModulus = 1U << 16;
        if (UNLIKELY(sqDepth_ == 0 || sqDepth_ > piModulus || (piModulus % sqDepth_) != 0)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] SQ depth[%u] is incompatible with u16 PI wrap", __func__, sqDepth_));
        }
        if (UNLIKELY(sqVa_ == 0 || (sqVa_ & (SQE_SIZE_64 - 1)) != 0)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] SQ base[%p] is null or not 64-byte aligned", __func__,
                reinterpret_cast<void *>(sqVa_)));
        }

        const u32 registeredCpuNum = ::GetCPUNum();
        batchStagingStats_.registeredCpuNum = registeredCpuNum;
        const u64 maxShardNum = static_cast<u64>(registeredCpuNum) * 2;
        if (UNLIKELY(parallelism > 1 && (registeredCpuNum <= 1
                                         || static_cast<u64>(parallelism) > maxShardNum))) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] scheduler cannot generate requested shards[%u], cpuCoreNum[%u], maxShards[%llu]",
                __func__, parallelism, registeredCpuNum, maxShardNum));
        }

        u32 wqeCount = 0;
        for (const auto &localSlice : loc) {
            if (localSlice.GetSize() != 0) {
                ++wqeCount;
            }
        }
        if (wqeCount == 0) {
            out.pi = pi;
            return;
        }
        if (UNLIKELY(wqeCount > sqDepth_)) {
            // ci/sqCiAddr is not exposed as a validated free-space query here; the caller must guarantee
            // that the requested slots are available before selecting this experiment path.
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] batch WQE count[%u] exceeds SQ depth[%u]", __func__, wqeCount, sqDepth_));
        }

        const u32 sqOffset = static_cast<u32>(pi) % sqDepth_;
        const u32 firstWqeCount = std::min(wqeCount, sqDepth_ - sqOffset);
        const u32 secondWqeCount = wqeCount - firstWqeCount;
        const u64 maxAddress = std::numeric_limits<u64>::max();
        auto checkSqSpan = [&](u32 spanSqOffset, u32 spanWqeCount) {
            if (spanWqeCount == 0) {
                return;
            }
            const u64 copySize = static_cast<u64>(spanWqeCount) * SQE_SIZE_64;
            const u64 byteOffset = static_cast<u64>(spanSqOffset) * SQE_SIZE_64;
            if (UNLIKELY(spanSqOffset >= sqDepth_ || spanWqeCount > sqDepth_ - spanSqOffset
                         || copySize > UINT32_MAX || sqVa_ > maxAddress - byteOffset
                         || copySize > maxAddress - (sqVa_ + byteOffset))) {
                THROW<InternalException>(StringFormat(
                    "[UbConnLite::%s] SQ span arithmetic overflow, sqOffset[%u], wqeCount[%u], sqDepth[%u]",
                    __func__, spanSqOffset, spanWqeCount, sqDepth_));
            }
        };
        checkSqSpan(sqOffset, firstWqeCount);
        checkSqSpan(0, secondWqeCount);

        constexpr u32 stagingAlignment = 64;
        const size_t stagingStorageSize = static_cast<size_t>(wqeCount) * sizeof(UdmaSqeWrite) + stagingAlignment - 1;
        const u64 stagingAllocationStartNs = GetCurAicpuTimestamp();
        std::vector<u8> stagingStorage(stagingStorageSize, 0);
        batchStagingStats_.stagingAllocationNs = GetCurAicpuTimestamp() - stagingAllocationStartNs;
        const u64 stagingPreparationStartNs = GetCurAicpuTimestamp();
        const u64 stagingStorageAddr = reinterpret_cast<u64>(stagingStorage.data());
        const u64 stagingAddr = (stagingStorageAddr + stagingAlignment - 1)
            & ~static_cast<u64>(stagingAlignment - 1);
        if (UNLIKELY((stagingAddr & (stagingAlignment - 1)) != 0)) {
            THROW<InternalException>(StringFormat("[UbConnLite::%s] staging address is not 64-byte aligned, addr[%p]",
                __func__, reinterpret_cast<void *>(stagingAddr)));
        }
        UdmaSqeWrite *staging = reinterpret_cast<UdmaSqeWrite *>(stagingAddr);
        for (u32 i = 0; i < wqeCount; ++i) {
            ::new (static_cast<void *>(staging + i)) UdmaSqeWrite;
        }

        const u64 timerProbeStartNs = GetCurAicpuTimestamp();
        const u64 timerProbeEndNs = GetCurAicpuTimestamp();
        batchStagingStats_.timerProbeNs = timerProbeEndNs - timerProbeStartNs;

        u64 descriptorIndex = 0;
        u32 stagingIndex = 0;
        while (descriptorIndex < loc.size()) {
            u32 builtWqeCount = 0;
            const u64 buildStartNs = GetCurAicpuTimestamp();
            while (builtWqeCount < stagingChunk && descriptorIndex < loc.size()) {
                if (loc[descriptorIndex].GetSize() == 0) {
                    ++descriptorIndex;
                    continue;
                }

                staging[stagingIndex] = {};
                const bool isLastDescriptor = (descriptorIndex == lastDescriptorIndex);
                SqeConfigLite wqeCfg = cfg;
                wqeCfg.cqeEn = isLastDescriptor;
                wqeCfg.placeOdr = isLastDescriptor ? UB_STRONG_ORDER : UB_RELAX_ORDER;
                wqeCfg.compOrder = isLastDescriptor ? 1 : 0;
                FillOneSqeWrite(loc[descriptorIndex], rmt[descriptorIndex], wqeCfg, &staging[stagingIndex],
                    UdmaSqOpcode::UDMA_OPC_READ, SlicePosition::ONLY);

                const u32 physicalOffset = static_cast<u32>(
                    (static_cast<u64>(sqOffset) + stagingIndex) % sqDepth_);
                staging[stagingIndex].comm.owner = (physicalOffset == sqDepth_ - 1) ? 1 : 0;
                ++stagingIndex;
                ++builtWqeCount;
                ++descriptorIndex;
            }
            batchStagingStats_.stagingBuildNs += GetCurAicpuTimestamp() - buildStartNs;
        }
        if (UNLIKELY(stagingIndex != wqeCount)) {
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] staging count mismatch, built[%u], expected[%u]", __func__, stagingIndex, wqeCount));
        }

        BatchSqCopySpan firstSpan;
        firstSpan.stagingIndex = 0;
        firstSpan.sqOffset = sqOffset;
        firstSpan.wqeCount = firstWqeCount;
        BatchSqCopySpan secondSpan;
        secondSpan.stagingIndex = firstWqeCount;
        secondSpan.sqOffset = 0;
        secondSpan.wqeCount = secondWqeCount;

        std::array<BatchSqCopyJob, BATCH_SQ_COPY_MAX_PARALLELISM> jobs{};
        auto assignSpan = [&](u32 jobBegin, u32 jobCount, const BatchSqCopySpan &span) {
            for (u32 i = 0; i < jobCount; ++i) {
                const u32 begin = static_cast<u32>(static_cast<u64>(span.wqeCount) * i / jobCount);
                const u32 end = static_cast<u32>(static_cast<u64>(span.wqeCount) * (i + 1) / jobCount);
                BatchSqCopyJob &job = jobs[jobBegin + i];
                job.spanCount = 1;
                job.spans[0].stagingIndex = span.stagingIndex + begin;
                job.spans[0].sqOffset = span.sqOffset + begin;
                job.spans[0].wqeCount = end - begin;
            }
        };
        if (secondWqeCount == 0) {
            assignSpan(0, parallelism, firstSpan);
        } else if (parallelism == 1) {
            jobs[0].spanCount = 2;
            jobs[0].spans[0] = firstSpan;
            jobs[0].spans[1] = secondSpan;
        } else {
            u32 firstJobCount = static_cast<u32>((static_cast<u64>(parallelism) * firstWqeCount
                + wqeCount / 2) / wqeCount);
            firstJobCount = std::max(1U, std::min(parallelism - 1, firstJobCount));
            assignSpan(0, firstJobCount, firstSpan);
            assignSpan(firstJobCount, parallelism - firstJobCount, secondSpan);
        }

        const u64 stagingPreparationElapsedNs = GetCurAicpuTimestamp() - stagingPreparationStartNs;
        batchStagingStats_.stagingPreparationNs = stagingPreparationElapsedNs >= batchStagingStats_.stagingBuildNs
            ? stagingPreparationElapsedNs - batchStagingStats_.stagingBuildNs : 0;

        // This array is local and explicitly aligned; it does not change UbConnLite's alignment requirement.
        alignas(64) std::array<BatchSqCopyJobResult, BATCH_SQ_COPY_MAX_PARALLELISM> jobResults{};
        batchStagingStats_.dispatchTid = diagnosticMode ? SalGetTid() : -1;
        // ParallelFor has no return status. The runtime join contract is a prerequisite; without it this path
        // is rejected before dispatch because a failed submission may leave a late callback using stack state.
        const auto work = [&](int64_t start, int64_t limit) {
            if (start < 0 || limit <= start || static_cast<u64>(start) >= parallelism
                || static_cast<u64>(limit) > parallelism) {
                return;
            }
            const u32 workerIndex = static_cast<u32>(start);
            jobResults[workerIndex].callbackEntered = true;
            BatchSqCopyWorkerStats *workerStats = diagnosticMode
                ? &batchStagingStats_.workerStats[workerIndex] : nullptr;
            if (workerStats != nullptr) {
                workerStats->entered = true;
            }
            u64 workerBytes = 0;
            s32 workerRet = 0;
            u32 completedJobCount = 0;
            try {
                if (workerStats != nullptr) {
                    workerStats->tid = SalGetTid();
                    workerStats->aicpuIndex = GetCurrentAicpuIndex();
                }
                for (int64_t jobIndex = start; jobIndex < limit; ++jobIndex) {
                    BatchSqCopyJobResult &jobResult = jobResults[static_cast<size_t>(jobIndex)];
                    jobResult.entered = true;
                    u64 jobBytes = 0;
                    u64 jobCalls = 0;
                    const BatchSqCopyJob &job = jobs[static_cast<size_t>(jobIndex)];
                    for (u32 spanIndex = 0; spanIndex < job.spanCount; ++spanIndex) {
                        const BatchSqCopySpan &span = job.spans[spanIndex];
                        if (span.wqeCount == 0) {
                            continue;
                        }
                        const u32 copyBytes = span.wqeCount * SQE_SIZE_64;
                        u8 *dst = reinterpret_cast<u8 *>(sqVa_ + static_cast<u64>(span.sqOffset) * SQE_SIZE_64);
                        const u8 *src = reinterpret_cast<const u8 *>(staging + span.stagingIndex);
                        const u64 copyStartNs = diagnosticMode ? GetCurAicpuTimestamp() : 0;
                        const s32 ret = memcpy_sp(dst, copyBytes, src, copyBytes);
                        const u64 copyEndNs = diagnosticMode ? GetCurAicpuTimestamp() : 0;
                        if (workerStats != nullptr) {
                            if (workerStats->startNs == 0) {
                                workerStats->startNs = copyStartNs;
                            }
                            workerStats->copyNs += copyEndNs - copyStartNs;
                            workerStats->endNs = copyEndNs;
                        }
                        if (UNLIKELY(ret != 0)) {
                            jobResult.ret = ret;
                            workerRet = ret;
                            break;
                        }
                        jobBytes += copyBytes;
                        ++jobCalls;
                    }
                    jobResult.bytes = jobBytes;
                    jobResult.calls = jobCalls;
                    workerBytes += jobBytes;
                    if (workerRet != 0) {
                        break;
                    }
                    ++completedJobCount;
                }
                if (workerRet != 0) {
                    if (workerStats != nullptr) {
                        workerStats->bytes = workerBytes;
                        workerStats->ret = workerRet;
                    }
                    return;
                }

                workerCompletionBarrier();
                if (workerStats != nullptr) {
                    workerStats->bytes = workerBytes;
                    workerStats->ret = workerRet;
                }
                for (u32 completedJob = 0; completedJob < completedJobCount; ++completedJob) {
                    jobResults[static_cast<size_t>(start) + completedJob].completed = true;
                }
            } catch (...) {
                const s32 workerFailure = workerRet == 0 ? -1 : workerRet;
                if (workerStats != nullptr) {
                    workerStats->bytes = workerBytes;
                    workerStats->ret = workerFailure;
                }
                for (int64_t jobIndex = start; jobIndex < limit; ++jobIndex) {
                    BatchSqCopyJobResult &jobResult = jobResults[static_cast<size_t>(jobIndex)];
                    jobResult.entered = true;
                    jobResult.ret = workerFailure;
                    jobResult.completed = false;
                }
            }
        };

        const u64 dispatchStartNs = GetCurAicpuTimestamp();
        ::ParallelFor(static_cast<int64_t>(parallelism), 1, work);
        batchStagingStats_.parallelDispatchJoinNs = GetCurAicpuTimestamp() - dispatchStartNs;

        u32 callbackCount = 0;
        for (u32 workerIndex = 0; workerIndex < parallelism; ++workerIndex) {
            if (jobResults[workerIndex].callbackEntered) {
                ++callbackCount;
            }
        }
        batchStagingStats_.callbackCount = callbackCount;
        u64 sqCopyBytes = 0;
        u64 sqCopyCalls = 0;
        for (u32 jobIndex = 0; jobIndex < parallelism; ++jobIndex) {
            const auto &jobResult = jobResults[jobIndex];
            if (UNLIKELY(!jobResult.entered || !jobResult.completed || jobResult.ret != 0)) {
                batchStagingStats_.parallelCopyFailed = true;
                THROW<InternalException>(StringFormat(
                    "[UbConnLite::%s] parallel SQ copy job failed, entered[%u], completed[%u], ret[%d]",
                    __func__, jobResult.entered, jobResult.completed, jobResult.ret));
            }
            sqCopyBytes += jobResult.bytes;
            sqCopyCalls += jobResult.calls;
        }
        const u64 expectedSqCopyBytes = static_cast<u64>(wqeCount) * SQE_SIZE_64;
        if (UNLIKELY(sqCopyBytes != expectedSqCopyBytes || sqCopyCalls == 0)) {
            batchStagingStats_.parallelCopyFailed = true;
            THROW<InternalException>(StringFormat(
                "[UbConnLite::%s] SQ copy result mismatch, bytes[%llu/%llu], calls[%llu]",
                __func__, sqCopyBytes, expectedSqCopyBytes, sqCopyCalls));
        }
        batchStagingStats_.sqCopyBytes = sqCopyBytes;
        batchStagingStats_.sqCopyCalls = sqCopyCalls;
        batchStagingStats_.bulkCopyWqeCount = wqeCount;

        const u64 publisherBarrierStartNs = GetCurAicpuTimestamp();
        // The provider contract, not a CPU atomic fence, establishes SQ visibility before PI publication.
        publisherBarrier();
        batchStagingStats_.publisherBarrierNs = GetCurAicpuTimestamp() - publisherBarrierStartNs;

        const u64 piCommitStartNs = GetCurAicpuTimestamp();
        if (static_cast<u64>(sqOffset) + wqeCount >= sqDepth_) {
            ++piDetourCount;
            ++batchStagingStats_.ringWrapCount;
        }
        pi = static_cast<u16>(static_cast<u32>(pi) + wqeCount);
        out.pi = pi;
        batchStagingStats_.piCommitNs = GetCurAicpuTimestamp() - piCommitStartNs;
    } catch (...) {
        pi = batchStartPi;
        piDetourCount = batchStartPiDetourCount;
        out.pi = batchStartPi;
        batchStagingStats_.parallelCopyFailed = true;
        throw;
    }
#endif
}

void UbConnLite::FinalizeBatchSqCopyMetrics()
{
    if (!batchStagingStats_.parallelCopyUsed || !batchStagingStats_.diagnosticMode) {
        return;
    }

    std::array<s32, BATCH_SQ_COPY_MAX_PARALLELISM> tids{};
    std::array<s32, BATCH_SQ_COPY_MAX_PARALLELISM> aicpuIndices{};
    std::array<std::pair<u64, s32>, BATCH_SQ_COPY_MAX_PARALLELISM * 2> overlapEvents{};
    u32 tidCount = 0;
    u32 aicpuIndexCount = 0;
    u32 overlapEventCount = 0;
    u64 minCopyStartNs = std::numeric_limits<u64>::max();
    u64 maxCopyEndNs = 0;

    for (u32 workerIndex = 0; workerIndex < batchStagingStats_.requestedParallelism; ++workerIndex) {
        const auto &workerStats = batchStagingStats_.workerStats[workerIndex];
        if (!workerStats.entered) {
            continue;
        }
        if (workerStats.tid == batchStagingStats_.dispatchTid) {
            batchStagingStats_.callerParticipated = true;
        }
        if (workerStats.bytes == 0 || workerStats.startNs >= workerStats.endNs) {
            continue;
        }

        ++batchStagingStats_.nonEmptyCallbackCount;
        batchStagingStats_.workerBusyNs += workerStats.copyNs;
        minCopyStartNs = std::min(minCopyStartNs, workerStats.startNs);
        maxCopyEndNs = std::max(maxCopyEndNs, workerStats.endNs);

        bool tidSeen = false;
        for (u32 index = 0; index < tidCount; ++index) {
            tidSeen = tidSeen || tids[index] == workerStats.tid;
        }
        if (!tidSeen && workerStats.tid >= 0) {
            tids[tidCount++] = workerStats.tid;
        }

        bool aicpuIndexSeen = false;
        for (u32 index = 0; index < aicpuIndexCount; ++index) {
            aicpuIndexSeen = aicpuIndexSeen || aicpuIndices[index] == workerStats.aicpuIndex;
        }
        if (!aicpuIndexSeen && workerStats.aicpuIndex >= 0) {
            aicpuIndices[aicpuIndexCount++] = workerStats.aicpuIndex;
        }

        overlapEvents[overlapEventCount++] = std::make_pair(workerStats.startNs, 1);
        overlapEvents[overlapEventCount++] = std::make_pair(workerStats.endNs, -1);
    }

    batchStagingStats_.activeCopyTidCount = tidCount;
    batchStagingStats_.activeCopyAicpuIndexCount = aicpuIndexCount;
    batchStagingStats_.sqCopyWallNs = maxCopyEndNs >= minCopyStartNs
        ? maxCopyEndNs - minCopyStartNs : 0;

    std::sort(overlapEvents.begin(), overlapEvents.begin() + overlapEventCount,
        [](const auto &lhs, const auto &rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
    s32 activeCopies = 0;
    for (u32 eventIndex = 0; eventIndex < overlapEventCount; ++eventIndex) {
        activeCopies += overlapEvents[eventIndex].second;
        batchStagingStats_.maxOverlap = std::max(batchStagingStats_.maxOverlap,
            static_cast<u32>(activeCopies));
    }
    batchStagingStats_.bulkCopyNs = batchStagingStats_.workerBusyNs;
    batchStagingStats_.bulkCopyCalls = batchStagingStats_.sqCopyCalls;
}

void UbConnLite::ReadReduce(ReduceIn reduceIn, const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt,
                            const StreamLite &stream, const SqeConfigLite &cfg, ConnLiteOperationOut &out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlices(loc, rmt, maxReadSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
            FillCommSqeReduceInfo(sqe.comm, reduceIn.reduceOp, reduceIn.dataType);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);
        },
        reduceIn.dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::Write(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                       const StreamLite &stream, ConnLiteOperationOut &out)
{
    HCCL_INFO("[UbConnLite::%s] start, loc size = %llu", __func__, loc.GetSize());

    ProcessSlices(loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
        UdmaSqeWrite sqe{};
        FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
        ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);
    });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::InlineWrite(const u8 *data, u16 size, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                             const StreamLite &stream, ConnLiteOperationOut &out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // 构造sqe
    UdmaSqeWrite sqe{};
    sqe.comm.inlineEn     = 1;
    sqe.comm.inlineMsgLen = size;
    FillCommSqe(&(sqe.comm), rmt, cfg, UdmaSqOpcode::UDMA_OPC_WRITE);
    auto ret = memcpy_sp(sqe.u.inlineData.data, SQE_INLINE_DATA_SIZE, data, size);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    }

    // 写wqe到va
    ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, ConnLiteOperationOut.datasize = %u, conn[%s]",
              __func__, out.pi, out.dataSize, Describe().c_str());
}

void UbConnLite::FillNotifySqe(struct UdmaSqeNotify *sqe, const RmtRmaBufSliceLite &notify, u64 notifyData) const
{
    sqe->notifyTokenId    = notify.GetTokenId();
    sqe->notifyTokenValue = notify.GetTokenValue();
    sqe->notifyAddrLow    = notify.GetAddr() & ADDR_BIT_LOW;
    sqe->notifyAddrHigh   = notify.GetAddr() >> ADDR_BIT_OFFSET;
    sqe->notifyDataLow    = notifyData & ADDR_BIT_LOW;
    sqe->notifyDataHigh   = notifyData >> ADDR_BIT_OFFSET;
    HCCL_INFO("UbConnLite FillNotifySqe sqe->notifyAddrLow = %u "
              "sqe->notifyAddrHigh = %u, sqe->notifyDataLow = %u, sqe->notifyDataHigh = %u",
              sqe->notifyAddrLow, sqe->notifyAddrHigh, sqe->notifyDataLow, sqe->notifyDataHigh);
}

void UbConnLite::FillLocalSgeSqe(UdmaNormalSge *sqe, const RmaBufSliceLite &loc) const
{
    sqe->length       = loc.GetSize();
    sqe->tokenId      = loc.GetTokenId();
    sqe->dataAddrLow  = loc.GetAddr() & ADDR_BIT_LOW;
    sqe->dataAddrHigh = loc.GetAddr() >> ADDR_BIT_OFFSET;
    HCCL_INFO("UbConnLite FillLocalSgeSqe sqe->length = %u, sqe->dataAddrLow = %u "
              "sqe->dataAddrHigh = %u",
              sqe->length, sqe->dataAddrLow, sqe->dataAddrHigh);
}

void UbConnLite::WriteReduce(DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite &loc,
                             const StreamLite &stream, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                             ConnLiteOperationOut &out)
{
    HCCL_INFO("[UbConnLite::%s] start, dataType = %u, reduceOp %u, loc.addr = %llu, "
              "rmt.addr = %llu, cfg.cqeEn = %u, out.pi = %u",
              __func__, dataType, reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi);

    ProcessSlices(loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteWithNotify(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                                 ConnLiteOperationOut &out, const RmtRmaBufSliceLite &notify, const StreamLite &stream,
                                 u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);
        },
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            ProcessOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData,
                                    WRITE_WITH_NOTIFY_OPCODE, slicePos, stream);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteReduceWithNotify(DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite &loc,
                                       const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg, const StreamLite &stream,
                                       ConnLiteOperationOut &out, const RmtRmaBufSliceLite &notify, u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);
        },
        [&](const RmaBufSliceLite &locSlice, const RmtRmaBufSliceLite &rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            ProcessOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData,
                                    WRITE_WITH_NOTIFY_OPCODE, slicePos, stream);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::CustomizeSqeByOneSidedComm(UdmaSqeCommon *sqe, bool isLastWqe) const
{
    /* 表示SQE是否需要上报CQE:为1表示此SQE需要上报CQE，为0表示不需要 */
    sqe->cqe = isLastWqe;

    /* 2’b00:No order，表示当前报文与其他报文无保序要求
       2’b01:Relax Order，表示当前报文与后续的Strong Order报文有保序要求，strong order报文不能超越relax order报文执行。
       2’b10：Strong Order，表示当前报文有保序要求，该报文与前面的Relax Order报文有保序要求。
       2’b11：Reserved。
    */
    sqe->placeOdr = (isLastWqe == true ? 0x02 : 0x01);

    /* ODR[2]表示请求报文在目的端的completion order属性，表示当前报文和前面报文是否存在completion序：
       1’b0 :no order，表示当前报文和前面报文没有completion序要求，报文对应的CQE可以乱序上报。
       1’b1 :表示当前报文和前面报文有completion序要求，报文对应的CQE需要保序上报
    */
    sqe->compOrder = 1;

    /* 表示是否使能fence保序。为1时表示使能，为0时表示不使能。对于send/write/atomic SQE
       当fence为1时需要等待前面所有read和Atomic完成才开始执行，即等待前面发出的read或Atomic接收到所有response
    */
    sqe->fence = (isLastWqe == true ? 0x01 : 0x00);

    HCCL_INFO(
        "UbConnLite CustomizeSqeByOneSidedComm sqe->cqe =%u, sqe->placeOdr = %u sqe->compOrder =%u, sqe->fence = %u",
        sqe->cqe, sqe->placeOdr, sqe->compOrder, sqe->fence);
}

void UbConnLite::FillBatchOneWqe(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                                 bool isLastWqe, u32 opCode, const StreamLite &stream)
{
    (void)stream;
    HCCL_INFO("UbConnLite FillBatchOneWqe start, loc[%s], rmt[%s]", loc.Describe().c_str(), rmt.Describe().c_str());

    u32 sqOffset = pi % sqDepth_;
    pi = pi + 1;
    if (UNLIKELY(pi > sqDepth_)) {
        pi = pi % sqDepth_;
    }

    // 写入wqe数据到out.data
    UdmaSqeWrite sqe{};
    sqe.comm.inlineEn = 0;
    FillCommSqe(&(sqe.comm), rmt, cfg, opCode);
    FillLocalSgeSqe(&(sqe.u.sge), loc);

    if (UNLIKELY(sqe.u.sge.length == 0)) {
        sqe.comm.sgeNum = 0;
    }

    CustomizeSqeByOneSidedComm(&(sqe.comm), isLastWqe);

    HCCL_INFO("UbConnLite BatchWrite cp data to va %llu, pi %u", sqVa_, pi);
    u8 *va = reinterpret_cast<u8 *>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (dwqeCacheLocked_ == false) {
        auto ret = memcpy_sp(va, SQE_SIZE_64, &sqe, sizeof(UdmaSqeWrite));
        if (UNLIKELY(ret != 0)) {
            HCCL_ERROR("UbConnLite::BatchWrite FillCommSqe memcpy failed, ret=%d", ret);
            THROW<InternalException>(StringFormat("UbConnLite::BatchWrite memcpy_sp failed, ret = %d", ret));
        }
    }
    HCCL_INFO("UbConnLite BatchWrite cp data to va end va(%p)", va);
}

void UbConnLite::BatchProcessOneSlice(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                                      u32 maxSliceSize, bool isLastSlice, u32 opCode, const StreamLite &stream)
{
    u64 dataSize = loc.GetSize();
    // 按照UDMA能力切分数据
    bool isLastWqe;
    u64  offset = 0;

    // 使用整数除法和取余运算优化循环
    u64 numIterations = dataSize / maxSliceSize;
    u64 remainingSize = dataSize % maxSliceSize;

    for (u64 i = 0; i < numIterations; ++i) {
        isLastWqe = false;
        if ((remainingSize == 0) && (i == numIterations - 1) && isLastSlice) {
            isLastWqe = true;
        }

        // 构造本次wqe的log和rmt RmaBufSilce
        RmaBufSliceLite    locTmp(loc.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(rmt.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, rmt.GetRkey(), rmt.GetTokenId(),
                                  rmt.GetTokenValue(), UINT32_MAX);

        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);

        offset += UB_DMA_MAX_READ_WEITE_SIZE;
    }

    // 处理剩余的数据
    if (remainingSize > 0 && isLastSlice) {
        isLastWqe = true;

        RmaBufSliceLite    locTmp(loc.GetAddr() + offset, remainingSize, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(rmt.GetAddr() + offset, remainingSize, rmt.GetRkey(), rmt.GetTokenId(),
                                  rmt.GetTokenValue(), UINT32_MAX);
        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);
    }
}

void UbConnLite::BatchCommDataProcess(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                                      const SqeConfigLite &cfg, u32 maxSliceSize, u32 opCode, const StreamLite &stream)
{
    u64 siliceSize = loc.size();
    // 按照UDMA能力切分数据, 组装wqe
    for (u64 i = 0; i < siliceSize; i++) {
        BatchProcessOneSlice(loc[i], rmt[i], cfg, maxSliceSize, (i == (siliceSize - 1)), opCode, stream);
    }

    return;
}

void UbConnLite::BatchOneSidedRead(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                                   const SqeConfigLite &cfg, const StreamLite &stream, ConnLiteOperationOut &out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxReadSize, UdmaSqOpcode::UDMA_OPC_READ, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchRead end, out.pi = %u", out.pi);
}

void UbConnLite::BatchOneSidedWrite(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                                    const SqeConfigLite &cfg, const StreamLite &stream, ConnLiteOperationOut &out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxWriteSize, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchWrite end, out.pi = %u", out.pi);
}

std::string UbConnLite::Describe()
{
    return StringFormat("UbConnLite[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
                        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, locEid=%s, rmtEid=%s,jettyPi=%u, jettyCi=%u]",
                        dieId_, funcId_, jettyId_, dbAddr_, sqVa_, sqDepth_, jfcPollMode_, tpn_, dwqeCacheLocked_,
                        Bytes2hex(locEid_.raw, sizeof(locEid_.raw)).c_str(), Bytes2hex(rmtEid_.raw, sizeof(rmtEid_.raw)).c_str(), 
                        pi, ci);
}

constexpr uint32_t UB_WQE_NUM_PER_SQE = 4; // URMA约束每个SQE包含4个WQEBB
UbConnLite::UbConnLite(const UbConnLiteParam &liteParam)
{
    wqeStagingChunk_ = WQE_STAGING_CHUNK_MAX;
    HCCL_INFO("[UbConnLite::%s] liteParam[%s]", __func__, liteParam.Describe().c_str());
    dieId_           = liteParam.dieId;
    funcId_          = liteParam.funcId;
    jettyId_         = liteParam.jettyId;
    dbAddr_          = liteParam.dbAddr;
    sqVa_            = liteParam.sqVa;
    // host侧创建jetty指定的sqDepth为sqeBBNum,device侧需要感知wqebbnum,URMA约束每个SQE包含4个WQEBB
    sqDepth_         = liteParam.sqDepth * UB_WQE_NUM_PER_SQE;
    dwqeCacheLocked_ = liteParam.dwqeCacheLocked;
    jfcPollMode_     = liteParam.jfcPollMode;
    tpn_             = liteParam.tpn;

    maxReadSize = liteParam.maxReadSize;
    maxWriteSize = liteParam.maxWriteSize;

    (void)memcpy_sp(rmtEid_.raw, URMA_EID_LEN, liteParam.rmtEid.raw, URMA_EID_LEN);
    (void)memcpy_sp(locEid_.raw, URMA_EID_LEN, liteParam.locEid.raw, URMA_EID_LEN);
    HCCL_INFO("%s", Describe().c_str());
}

UbConnLite::UbConnLite(const UbJettyLiteId &id, const UbJettyLiteAttr &attr, const Eid &rmtInfo)
    : RmaConnLite(id, attr, rmtInfo),
      maxReadSize(UB_DMA_MAX_READ_WEITE_SIZE),
      maxWriteSize(UB_DMA_MAX_READ_WEITE_SIZE),
      wqeStagingChunk_(WQE_STAGING_CHUNK_MAX)
{
}

std::string UbConnLiteParam::Describe() const
{
     return StringFormat("UbConnLiteParam[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
                        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, sqCiAddr=0x%llx, rmtEid=%s, localEid=%s, "
                        "maxReadSize=%u, maxWriteSize=%u]",
                        dieId, funcId, jettyId, dbAddr, sqVa, sqDepth, jfcPollMode, tpn, dwqeCacheLocked, sqCiAddr,
                        Bytes2hex(rmtEid.raw, sizeof(rmtEid.raw)).c_str(), Bytes2hex(locEid.raw, sizeof(locEid.raw)).c_str(),
                        maxReadSize, maxWriteSize);
}

UbConnLiteParam::UbConnLiteParam(std::vector<char> &uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> dieId;
    binaryStream >> funcId;
    binaryStream >> jettyId;

    binaryStream >> jfcPollMode;
    binaryStream >> dwqeCacheLocked;
    binaryStream >> dbAddr;
    binaryStream >> sqCiAddr;
    binaryStream >> sqVa;
    binaryStream >> sqDepth;
    binaryStream >> tpn;
    binaryStream >> rmtEid.raw;
    binaryStream >> locEid.raw;
    binaryStream >> maxReadSize;
    binaryStream >> maxWriteSize;

    static auto lastPrintTime = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
    if (UNLIKELY(duration >= MAX_LOG_TIMEOUT_MS)) { 
        HCCL_INFO("%s", Describe().c_str());
        lastPrintTime = now;
    }
    HCCL_INFO("[UbConnLiteParam::%s] locEid[%s], rmtEid[%s]", __func__, locEid.Describe().c_str(), rmtEid.Describe().c_str());
}

} // namespace Hccl
