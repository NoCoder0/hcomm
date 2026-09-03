/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_CONN_LITE_H_
#define HCCLV2_UB_CONN_LITE_H_

#include <array>
#include <functional>
#include <queue>
#include <vector>
#include "data_type.h"
#include "reduce_op.h"
#include "rma_buf_slice_lite.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_conn_lite.h"
#include "udma_data_struct.h"
#include "kernel_param_lite.h"
#include "stream_lite.h"

// Enable only after the runtime callback-join and SQ visibility contracts are provided.
#ifndef HCOMM_ENABLE_AICPU_PARALLEL_SQ_COPY
#define HCOMM_ENABLE_AICPU_PARALLEL_SQ_COPY 0
#endif

namespace Hccl {

class UbTransportLiteImpl;

// Provider callbacks must be thread-safe, non-throwing, and establish the documented device visibility contract.
using BatchSqCopyBarrier = void (*)();

enum class SlicePosition { ONLY = 0, FIRST = 1, MIDDLE = 2, LAST = 3 };

struct UbConnLiteParam {
    u32 dieId;
    u32 funcId;
    u32 jettyId;

    u64  dbAddr;
    u64  sqVa;
    u32  sqDepth;
    u32  tpn;
    bool dwqeCacheLocked;
    u32  jfcPollMode; // 0代表STARS POLL， 1代表软件Poll
    u64  sqCiAddr;    // 预留给 软件poll CQ 的Jetty使用

    Eid rmtEid;
    Eid locEid;

    u32 maxReadSize;
    u32 maxWriteSize;

    UbConnLiteParam(std::vector<char> &uniqueId);

    std::string Describe() const;
};

class UbConnLite : public RmaConnLite {
public:
    friend class UbTransportLiteImpl;

    UbConnLite(const UbJettyLiteId &id, const UbJettyLiteAttr &attr, const Eid &rmtInfo);

    explicit UbConnLite(const UbConnLiteParam &liteParam);

    std::string Describe() final;

    void FillCommSqe(UdmaSqeCommon *sqe, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg, u32 opCode,
                     SlicePosition slicePos = SlicePosition::ONLY);

    void Read(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
              const StreamLite &stream, ConnLiteOperationOut &out) override;

    void ReadReduce(ReduceIn reduceIn, const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt,
                    const StreamLite &stream, const SqeConfigLite &cfg, ConnLiteOperationOut &out) override;

    void Write(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
               const StreamLite &stream, ConnLiteOperationOut &out) override;

    void InlineWrite(const u8 *data, u16 size, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                     const StreamLite &stream, ConnLiteOperationOut &out) override;

    void WriteReduce(DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite &loc, const StreamLite &stream,
                     const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg, ConnLiteOperationOut &out) override;

    void FillNotifySqe(struct UdmaSqeNotify *sqe, const RmtRmaBufSliceLite &notify, u64 notifyData) const;
    void FillLocalSgeSqe(UdmaNormalSge *sqe, const RmaBufSliceLite &loc) const;

    void WriteWithNotify(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                         ConnLiteOperationOut &out, const RmtRmaBufSliceLite &notify, const StreamLite &stream,
                         u64 notifyData) override;

    void WriteReduceWithNotify(DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite &loc,
                               const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg, const StreamLite &stream,
                               ConnLiteOperationOut &out, const RmtRmaBufSliceLite &notify, u64 notifyData) override;

    void CustomizeSqeByOneSidedComm(UdmaSqeCommon *sqe, bool isLastWqe) const;

    void FillBatchOneWqe(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                         bool isLastWqe, u32 opCode, const StreamLite &stream);

    void BatchProcessOneSlice(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                              u32 maxSliceSize, bool isLastSlice, u32 opCode, const StreamLite &stream);

    void BatchCommDataProcess(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                              const SqeConfigLite &cfg, u32 maxSliceSize, u32 opCode, const StreamLite &stream);

    void BatchOneSidedRead(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                           const SqeConfigLite &cfg, const StreamLite &stream, ConnLiteOperationOut &out) override;
    void BatchOneSidedWrite(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                            const SqeConfigLite &cfg, const StreamLite &stream, ConnLiteOperationOut &out) override;
private:
    static constexpr u32 WQE_STAGING_CHUNK_MAX = 32;
    static constexpr u32 BATCH_SQ_COPY_MAX_PARALLELISM = 16;

    struct BatchSqCopyWorkerStats {
        u64 bytes{0};
        u64 copyNs{0};
        u64 startNs{0};
        u64 endNs{0};
        s32 ret{0};
        s32 tid{-1};
        s32 aicpuIndex{-1};
        bool entered{false};
    };

    struct BatchStagingStats {
        u64 stagingBuildNs{0};
        u64 stagingAllocationNs{0};
        u64 stagingPreparationNs{0};
        u64 bulkCopyNs{0};
        u64 bulkCopyCalls{0};
        u64 bulkCopyWqeCount{0};
        u64 ringWrapCount{0};
        u64 timerProbeNs{0};
        u64 parallelDispatchJoinNs{0};
        u64 sqCopyWallNs{0};
        u64 workerBusyNs{0};
        u64 sqCopyBytes{0};
        u64 sqCopyCalls{0};
        u64 piCommitNs{0};
        u64 publisherBarrierNs{0};
        u32 stagingChunk{WQE_STAGING_CHUNK_MAX};
        u32 requestedParallelism{1};
        u32 registeredCpuNum{0};
        u32 callbackCount{0};
        u32 nonEmptyCallbackCount{0};
        u32 activeCopyTidCount{0};
        u32 activeCopyAicpuIndexCount{0};
        u32 maxOverlap{0};
        s32 dispatchTid{-1};
        bool stagingUsed{false};
        bool parallelCopyUsed{false};
        bool parallelCopyFailed{false};
        bool callerParticipated{false};
        bool diagnosticMode{false};
        std::array<BatchSqCopyWorkerStats, BATCH_SQ_COPY_MAX_PARALLELISM> workerStats{};

        u64 BulkCopyAvgNsPerWqe() const
        {
            return bulkCopyWqeCount == 0 ? 0 : bulkCopyNs / bulkCopyWqeCount;
        }
    };

    u16  pi{0};
    u16  ci{0};
    u32  piDetourCount{0};
    u32  ciDetourCount{0};
    u32  maxReadSize{0};
    u32  maxWriteSize{0};
    u32  wqeStagingChunk_{WQE_STAGING_CHUNK_MAX};
    BatchStagingStats batchStagingStats_{};

    static bool IsLegalWqeStagingChunk(u32 chunk);

    u32 GetWqeStagingChunk() const
    {
        return wqeStagingChunk_;
    }

    // ExecuteBatchTransfer 专用的全 READ 批量构造入口，不扩展 HCOMM 对外接口。
    void BatchRead(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                   const SqeConfigLite &cfg, u32 lastDescriptorIndex, const StreamLite &stream,
                   ConnLiteOperationOut &out, u32 stagingChunk, bool parallelCopy, u32 parallelism,
                   BatchSqCopyBarrier workerCompletionBarrier, BatchSqCopyBarrier publisherBarrier,
                   bool schedulerJoinGuaranteed, bool diagnosticMode);

    void BatchReadParallel(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite> &rmt,
                           const SqeConfigLite &cfg, u32 lastDescriptorIndex, const StreamLite &stream,
                           ConnLiteOperationOut &out, u32 stagingChunk, u32 parallelism,
                           BatchSqCopyBarrier workerCompletionBarrier, BatchSqCopyBarrier publisherBarrier,
                           bool schedulerJoinGuaranteed, bool diagnosticMode);

    void FinalizeBatchSqCopyMetrics();

    const BatchStagingStats &GetBatchStagingStats() const
    {
        return batchStagingStats_;
    }
    void ProcessSlices(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, u32 maxSliceSize,
        std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSlice,
        DataType dataType = DataType::INVALID) const;
    void ProcessSlicesWithNotify(
        const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, u32 maxSliceSize,
        std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSlice,
        std::function<void(const RmaBufSliceLite &, const RmtRmaBufSliceLite &, SlicePosition)> processOneSliceWithNotify,
        DataType                                                                 dataType = DataType::INVALID) const;
    void ProcessOneWqe(UdmaSqeWrite *sqe, UdmaSqOpcode opCode, const StreamLite &stream);
    void ProcessOneWqeWithNotify(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                                 UdmaSqeWriteWithNotify *sqe, const RmtRmaBufSliceLite &notify, u64 notifyData,
                                 u32 opCode, SlicePosition slicePos, const StreamLite &stream);
    void FillCommSqeReduceInfo(UdmaSqeCommon &sqeComm, ReduceOp reduceOp, DataType dataType, u32 udfType = 0) const;
    void FillOneSqeWrite(const RmaBufSliceLite &loc, const RmtRmaBufSliceLite &rmt, const SqeConfigLite &cfg,
                         UdmaSqeWrite *sqe, UdmaSqOpcode opCode, SlicePosition slicePos);
    void MemorySetAndCopy(u8 *va, u32 sqeSize, void *sqe);
};
} // namespace Hccl

#endif // HCCLV2_UB_CONN_LITE_H_
