/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UB_MEM_TRANSPORT_LITE_H
#define UB_MEM_TRANSPORT_LITE_H

#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include "base_transport_lite_impl.h"
#include "notify_lite.h"
#include "task_param.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_conn_lite.h"
#include "kernel_param_lite.h"
#include "hcomm_primitives.h"

namespace Hccl {

class UbTransportLiteImpl : public BaseTransportLiteImpl {
public:
    explicit UbTransportLiteImpl(std::vector<char>                                                 &uniqueId,
                                 std::function<void(u32 streamId, u32 taskId, const TaskParam &taskParam)> callback);

    UbTransportLiteImpl(std::vector<char> &uniqueId);
    void Init(std::vector<char> &uniqueId);

    ~UbTransportLiteImpl() override;

    std::string Describe() const override;

    Buffer GetRmtBuffer(u32 index) override;

    Eid GetLocEid() const;
    Eid GetRmtEid() const;

    void Post(u32 index, const StreamLite &stream) override;

    void Wait(u32 index, const StreamLite &stream) override;

    void WaitWithTimeout(u32 index, const StreamLite &stream, u32 timeout) override;

    void Read(const RmaBufferLite &loc, const Buffer &rmt, const StreamLite &stream) override;

    void Write(const RmaBufferLite &loc, const Buffer &rmt, const StreamLite &stream) override;

    void ReadReduce(const RmaBufferLite &loc, const Buffer &rmt, const ReduceIn &reduceIn,
                    const StreamLite &stream) override;

    void WriteReduce(const RmaBufferLite &loc, const Buffer &rmt, const ReduceIn &reduceIn,
                     const StreamLite &stream) override;

    void WriteWithNotify(const RmaBufferLite &loc, const Buffer &rmt, const WithNotifyIn &withNotify,
                         const StreamLite &stream) override;

    void WriteReduceWithNotify(const RmaBufferLite &loc, const Buffer &rmt, const ReduceIn &reduceIn,
                               const WithNotifyIn &withNotify, const StreamLite &stream) override;

    void BatchOneSidedWrite(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite>  &rmt,
        const StreamLite &stream) override;

    void BatchOneSidedRead(const vector<RmaBufSliceLite> &loc, const vector<RmtRmaBufSliceLite>  &rmt,
        const StreamLite &stream) override;
    
    void BatchTransfer(const std::vector<RmaBufferLite> &loc, const std::vector<Buffer> &rmt,
                        const std::vector<TransferOp> &transferOp, const StreamLite &stream) override;
    // 子类独有方法，支持所有操作类型，用于aicpu场景批量下发任务
    void BatchTransferAll(const std::vector<RmaBufferLite> &loc, const std::vector<Buffer> &rmt,
                        const std::vector<TransferOp> &transferOp, const std::vector<uint32_t> &notifyIdxs, const StreamLite &stream);

    HcclResult BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite &rmaBufferLite) override;
    HcclResult Fence();

    HcclResult Clean();
    HcclResult Resume(std::vector<char> &uniqueId);
    void SetTaskExceptionEnable(bool flag) { taskExceptionEnable_ = flag; }

    HcclResult ExecuteBatchTransfer(StreamLite *streamLitePtr, const HcommBatchTransferDesc *transferDescs,
                        uint32_t transferDescNum);
private:
    u32 notifyNum{0};
    u32 bufferNum{0};
    u32 rmtbufferNum{0};
    u32 connNum{0};
    bool fence_{false};
    bool taskExceptionEnable_{true};

    struct RmtUbBufLite {
        u64         addr;
        u64         size;
        u32         tokenId;
        u32         tokenValue;
        u32         notifyId;
        std::string Describe() const
        {
            return StringFormat("RmtUbBufLite[addr=0x%llx, size=%llu, notifyId=%u]", addr, size, notifyId);
        }
    };

     struct LocUbBufLite {
        u64         addr;
        u64         size;
        u32         tokenId;
        u32         tokenValue;
        std::string Describe() const
        {
            return StringFormat("LocUbBufLite[addr=0x%llx, size=%llu]", addr, size);
        }
    };

    std::vector<char>    wqeData; // connection返回的WQE内容
    ConnLiteOperationOut connOut; // connection的输出

    void ClearConnOut();

    using RmtUbBufLiteVec = std::vector<RmtUbBufLite>;
    using RmtUbBufLiteMap = std::map<uintptr_t, RmtUbBufLite>;
    using LocUbBufLiteMap = std::map<uintptr_t, LocUbBufLite>;
    MAKE_ENUM(RmaUbBufType, NOTIFY, BUFFER)
    RmtUbBufLiteVec rmtNotifyVec;
    RmtUbBufLiteVec rmtBufferVec;
    RmtUbBufLiteMap rmtBufferMap; // 性能优化使用
    LocUbBufLiteMap locBufferMap;

    RmtRmaBufSliceLite GetRmtNotifySliceLite(u32 index);
    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(const Buffer &rmtBuf);

    RmaBufSliceLite GetRmaBufSlicelite(const RmaBufferLite &lite) const;
    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(const RmaBufferLite &lite) const;

    std::vector<std::unique_ptr<NotifyLite>> locNotifyVec;

    // N秒快恢需要清理的两个资源
    std::vector<std::vector<char>> connUniqueIdVec;
    std::vector<RmaConnLite *> connVec;

    std::function<void(u32 streamId, u32 taskId, const TaskParam &taskParam)> callback_{nullptr};

    void ProfilingProcess(void *src, void *dst, u64 size, const StreamLite &stream, DmaOp dmaOp,
                            u32 taskId);

    void ReduceProfilingProcess(void *src, void *dst, u64 size, const ReduceIn &reduceIn,
                                      const StreamLite &stream, u32 taskId);

    void ParseLocNotifyVec(std::vector<char> &data);

    void ParseRmtBufferVec(std::vector<char> &data, RmaUbBufType rmtType);
 
    void ParseLocBufferMap(std::vector<char> &data);

    void ParseConnVec(std::vector<char> &data);

    void BuildUbDbSendTask(const StreamLite &stream, const UbJettyLiteId &jettyLiteId, u32 pi);

    void BuildNotifyWaitTask(const StreamLite &stream, u32 notifyId);

    void CheckConnVec(const std::string &desc);

    void SetFenceConfig(SqeConfigLite &cfg);

    bool IsReportTask();

    void ExecProfiling(const std::vector<RmaBufferLite> &loc, const std::vector<Buffer> &rmt, 
                 const std::vector<BaseTransportLiteImpl::TransferOp> &transferOp, const StreamLite &stream, u32 taskId);

    inline void AddTaskCallback(const StreamLite &stream, u32 taskId, const TaskParam &taskParam)
    {
        if (callback_ != nullptr) {
            callback_(stream.GetSqId(), taskId, taskParam);
        }

        if (newCallback_ != nullptr) {
            newCallback_(stream.GetSqId(), taskId, taskParam, reinterpret_cast<u64>(this));
        }
    }

    inline void FillTaskParamDmaPub(TaskParam &taskParam, void *dst, u64 size, DmaOp dmaOp) const
    {
        taskParam.taskPara.DMA.dst      = dst;
        taskParam.taskPara.DMA.size     = size;
        taskParam.taskPara.DMA.notifyID = INVALID_VALUE_NOTIFYID;
        taskParam.taskPara.DMA.notifyValue = 0xffffffff;
        taskParam.taskPara.DMA.linkType = DfxLinkType::UB;
        taskParam.taskPara.DMA.dmaOp    = dmaOp;
        taskParam.taskPara.DMA.locEid = GetLocEid();
        taskParam.taskPara.DMA.rmtEid = GetRmtEid();
    }

    void ExecProfilingAll(const std::vector<RmaBufferLite> &loc, const std::vector<Buffer> &rmt, 
                const std::vector<BaseTransportLiteImpl::TransferOp> &transferOp, const StreamLite &stream, u32 taskId,
                const std::vector<uint32_t> &notifyIdxs);
    void WriteWithNotifyProfilingProcess(void *src, void *dst, u64 size, const StreamLite &stream,
                                        u32 taskId, u64 notifyId);
    void WriteReduceWithNotifyProfilingProcess(void *src, void *dst, u64 size,
                                            const ReduceIn &reduceIn, const StreamLite &stream, u32 taskId, u64 notifyId);
    void NotifyRecordProfilingProcess(void *dst, u64 size,
                                    const StreamLite &stream, u32 taskId, u64 notifyId);
};

} // namespace Hccl
#endif
