/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mutex>
#include <cstring>
#include <memory>
#include <vector>

#include "hccl/hccl_res.h"
#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_result_defs.h"
#include "log.h"
#include "hcomm_c_adpt.h"
#include "hcom_common.h"
#include "endpoint.h"
#include "thread.h"
#include "aicpu_ts_thread.h"
#include "cpu_ts_thread.h"
#include "aicpu_ts_urma_channel.h"
#include "mem_device_pub.h"
#include "channel_param.h"
#include "launch_aicpu.h"
#include "comm_configer.h"
#include "env_config/env_config.h"
#include "endpoint_map.h"
#ifdef ENABLE_EXPERIMENTAL
#include "nic_plugin_dispatcher.h"
#endif

#include "../hcomm_res_mgr.h"

#include "param_check_pub.h"
#include "comm_engine_utils.h"

#include "exception_handler.h"
#include "hcclCommDfx.h"
#include "hcclCommOp.h"
#include "channel_process.h"
#include "launch_device.h"
#include "endpoint_monitor.h"
#include "hcomm_adapter_runtime.h"
#include "adapter_rts_common.h"
#include "host_mode_detector.h"


namespace hcomm {
static std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>> g_ThreadMap;
static aclrtBinHandle g_BinHandle;
static std::mutex g_BinHandleMtx;
}  // namespace hcomm

using namespace hcomm;
static HcommEndpointMap g_EndpointMap;

namespace {
HcclResult RefreshCurrentDeviceContext()
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId, true));
    HCCL_INFO("[RefreshCurrentDeviceContext] deviceLogicId[%d], devicePhyId[%u].", deviceLogicId, devicePhyId);
    return HCCL_SUCCESS;
}

HcclResult RefreshEndpointContext(const EndpointDesc &endpointDesc)
{
    if (endpointDesc.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        return HCCL_SUCCESS;
    }
    return RefreshCurrentDeviceContext();
}

HcclResult RefreshCommEngineContext(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        return HCCL_SUCCESS;
    }
    return RefreshCurrentDeviceContext();
}
}

namespace {
HcclResult ValidateEndpointDesc(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    CHK_PTR_NULL(endpoint);
    CHK_PTR_NULL(endpointHandle);
    if (endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE && endpoint->loc.locType != ENDPOINT_LOC_TYPE_HOST) {
        HCCL_ERROR("[%s] Only support END_POINT_LOCATION_DEVICE AND END_POINT_LOCATION_HOST, but "
                   "endpoint->loc.locType is %d",
            __func__,
            endpoint->loc.locType);
        return HCCL_E_PARA;
    }
    // 防止 host-only 模式误配 DEVICE loc，或 NPU 模式误配 HOST loc 而后走错路径。
    // HostOnly 下 DEVICE loc 必然失败；NPU 模式下 HOST loc 不被允许（应通过 DEVICE loc 触发 NPU 路径）。
    const bool hostOnly = hccl::HostModeDetector::IsHostOnly();
    if (hostOnly && endpoint->loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_ERROR("[%s] host-only mode detected but ENDPOINT_LOC_TYPE_DEVICE requested; "
                   "use ENDPOINT_LOC_TYPE_HOST or set HCOMM_HOST_ONLY=0.", __func__);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateCommEngineAgainstHostMode(CommEngine engine)
{
    const bool hostOnly = hccl::HostModeDetector::IsHostOnly();
    if (hostOnly && (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS)) {
        HCCL_ERROR("[%s] host-only mode detected but AICPU engine[%d] requested; "
                   "use COMM_ENGINE_CPU.", __func__, static_cast<int>(engine));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDeviceEndpointMonitorIfNeeded(const EndpointDesc *endpoint, EndpointHandle handle)
{
    if ((endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE) ||
        ((endpoint->protocol != COMM_PROTOCOL_UBC_CTP) && (endpoint->protocol != COMM_PROTOCOL_UBC_TP))) {
        return HCCL_SUCCESS;
    }

    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(devLogicIdSigned < 0,
        HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned), HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RegisterToEndpointMonitor(devLogicIdSigned, handle);
    return HCCL_SUCCESS;
}

HcclResult CreateBuiltinEndpoint(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
	CHK_RET(RefreshEndpointContext(*endpoint));
    std::unique_ptr<Endpoint> endpointPtr = nullptr;
    HcclResult ret = Endpoint::CreateEndpoint(*endpoint, endpointPtr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call Endpoint::CreateEndpoint failed");
        return ret;
    }
    CHK_PTR_NULL(endpointPtr);
    ret = endpointPtr->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("call endpointPtr->Init failed");
        return ret;
    }

    const EndpointHandle handle = reinterpret_cast<EndpointHandle>(endpointPtr.get());
    CHK_PTR_NULL(handle);
    EXCEPTION_CATCH(g_EndpointMap.AddEndpoint(handle, std::move(endpointPtr)), return HCCL_E_INTERNAL);
    *endpointHandle = handle;
    CHK_RET(RegisterDeviceEndpointMonitorIfNeeded(endpoint, handle));
    HCCL_INFO("[%s] endpointDesc.protocol [%d] and endpointDesc.loc.locType [%d] create endpointHandle [%p] done.",
        __func__, endpoint->protocol, endpoint->loc.locType, handle);
    return HCCL_SUCCESS;
}
}

HcommResult CheckUbAttr(HcommChannelDesc &channelDesc)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBC_TP
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBOE
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBG
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBC_CTP) {
        return HCCL_SUCCESS;
    }

    // check sqDepth
    if (channelDesc.ubAttr.sqDepth == 0xFFFFFFFF) { // 0xFFFFFFFF表示使用默认值
        HCCL_INFO("[%s] use default ubAttr.sqDepth.", __func__);
        return HCCL_SUCCESS;
    }

    // sqDepth的合理范围在[16, 256]
    if (channelDesc.ubAttr.sqDepth < 16 || channelDesc.ubAttr.sqDepth > 256) {
        HCCL_ERROR("[%s] invalid ubAttr.sqDepth[%u], should be 0 or >= 16 and <= 256.", __func__, channelDesc.ubAttr.sqDepth);
        return HCCL_E_PARA;
    }

    // channelDesc.ubAttr.sqDepth调整到2的整数次幂
    auto GetNextPowerOfTwo = [](uint32_t n) -> uint32_t {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    };

    channelDesc.ubAttr.sqDepth = GetNextPowerOfTwo(channelDesc.ubAttr.sqDepth);

    return HCCL_SUCCESS;
}

HcommResult CheckRoceAttr(HcommChannelDesc &channelDesc)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_ROCE) {
        return HCCL_SUCCESS;
    }

    if (channelDesc.roceAttr.queueNum == INVALID_UINT) {
        channelDesc.roceAttr.queueNum = 1;
        HCCL_INFO("[%s] set roceAttr.queueNum to 1.", __func__);
    }

    return HCCL_SUCCESS;
}

namespace {
void ApplyHcommChannelDescV1Fields(const HcommChannelDesc &channelDesc, HcommChannelDesc &channelDescFinal)
{
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION_ONE) {
        return;
    }

    channelDescFinal.remoteEndpoint = channelDesc.remoteEndpoint;
    channelDescFinal.notifyNum = channelDesc.notifyNum;
    channelDescFinal.exchangeAllMems = channelDesc.exchangeAllMems;
    channelDescFinal.memHandles = channelDesc.memHandles;
    channelDescFinal.memHandleNum = channelDesc.memHandleNum;
    channelDescFinal.socket = channelDesc.socket;
    channelDescFinal.role = channelDesc.role;
    channelDescFinal.port = channelDesc.port;
}

HcommResult ProcessHcommChannelDescs(const HcommChannelDesc &channelDesc, HcommChannelDesc &channelDescFinal)
{
    if (channelDesc.header.size < sizeof(CommAbiHeader)) {
        HCCL_ERROR("[%s] invalid channelDesc.header.size[%u].", __func__, channelDesc.header.size);
        return HCCL_E_PARA;
    }

    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR("[%s] channelDesc.header.magicWord[0x%08x] is invalid, expected[0x%08x].",
            __func__, channelDesc.header.magicWord, channelDescFinal.header.magicWord);
        return HCCL_E_PARA;
    }

    const uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ?
        channelDescFinal.header.size : channelDesc.header.size) - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(reinterpret_cast<uint8_t *>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        reinterpret_cast<const uint8_t *>(&channelDesc) + sizeof(CommAbiHeader), copySize));
    ApplyHcommChannelDescV1Fields(channelDesc, channelDescFinal);
    if (channelDesc.header.version > HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING("The version of provided [%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING("The version of provided [%u] is lower than the current version[%u], "
            "configurations supported by later versions will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    }

    // qos：低版本时置默认值
    if (channelDesc.header.version <= HCOMM_CHANNEL_VERSION_ONE) {
        channelDescFinal.qos = 0xFFFFFFFFU;
    } else {
        channelDescFinal.qos = channelDesc.qos;
    }

    // v3：channelName，低版本时置 NULL
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        channelDescFinal.channelName = nullptr;
    } else {
        channelDescFinal.channelName = channelDesc.channelName;
        if (channelDescFinal.channelName != nullptr &&
            reinterpret_cast<uintptr_t>(channelDescFinal.channelName) == static_cast<uintptr_t>(-1)) {
            channelDescFinal.channelName = nullptr;
        }
    }

    if (channelDescFinal.channelName != nullptr) {
        size_t nameLen = strnlen(channelDescFinal.channelName, HCOMM_CHANNEL_NAME_MAX_LEN + 1);
        if (nameLen > HCOMM_CHANNEL_NAME_MAX_LEN) {
            HCCL_ERROR("[%s] channelName too long, max len[%u].", __func__, HCOMM_CHANNEL_NAME_MAX_LEN);
            return HCCL_E_PARA;
        }
    }

    return HCOMM_SUCCESS;
}

HcommResult NormalizeHcommChannelDescs(HcommChannelDesc *channelDescs, uint32_t channelNum,
    std::vector<HcommChannelDesc> &channelDescFinals)
{
    channelDescFinals.clear();
    channelDescFinals.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HcommChannelDesc channelDescFinal{};
        HcommResult ret = HcommChannelDescInit(&channelDescFinal, 1);
        if (ret != HCOMM_SUCCESS) {
            return ret;
        }
        ret = ProcessHcommChannelDescs(channelDescs[idx], channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] failed to normalize channelDesc[%u], ret[%d].", __func__, idx, ret);
            return ret;
        }
        ret = CheckUbAttr(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckUbAttr failed, ret[%d].", __func__, ret);
            return ret;
        }
        ret = CheckRoceAttr(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckRoceAttr failed, ret[%d].", __func__, ret);
            return ret;
        }

        channelDescFinals.push_back(channelDescFinal);
    }
    return HCOMM_SUCCESS;
}

} // namespace

HcommResult HcommResMgrInit(uint32_t devPhyId)
{
    bool noDevice = false;
      if (devPhyId == UINT32_MAX) {
          CHK_RET(ResolveRuntimeDevicePhyId(devPhyId, noDevice));
      }

    // 临时方案：触发统一平台层单例触发静态对象声明
    // 内部流程触发各种单例声明，保证时序
    EXCEPTION_HANDLE_BEGIN
    if (noDevice) {
        (void)HcommResMgr::GetInstance(devPhyId);
        return HCCL_SUCCESS;
    }

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        (void)HcommResMgr::GetInstance(devPhyId);
        return HcclResult::HCCL_SUCCESS;
    }());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

static HcclResult EnsureKernelBinLoaded(CommEngine engine) {
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        HCCL_INFO("[%s] engine[%s] kernel loading not required", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(hcomm::g_BinHandleMtx);
    if (g_BinHandle != nullptr) {
        return HCCL_SUCCESS;
    }
    std::string jsonPath;
    CHK_RET(hccl::GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";

    HcclResult ret = hccl::LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, g_BinHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[EnsureKernelBinLoaded] load aicpu file fail, path[%s]", jsonPath.c_str()),
                ret);
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointGet(EndpointHandle endpointHandle, void **endpoint)  // 根据endpointHandle返回Endpoint对象指针
{
    CHK_PTR_NULL(endpoint);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointGet(endpointHandle, endpoint, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto it = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(it == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);

    *endpoint = static_cast<void *>(it);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx] endpoint[0x%llx].",__func__, endpointHandle, endpoint);
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle)
{
    EXCEPTION_HANDLE_BEGIN
    (void)HcommResMgrInit();
    CHK_RET(ValidateEndpointDesc(endpoint, endpointHandle));
#ifdef ENABLE_EXPERIMENTAL
    bool pluginHandled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointCreate(endpoint, endpointHandle, pluginHandled)));
    if (pluginHandled) {
        HCCL_INFO("[NicPluginDebug][%s] plugin endpoint created, protocol[%d], handle[%p].",
            __func__, endpoint->protocol, *endpointHandle);
        return HCCL_SUCCESS;
    }
#endif
    CHK_RET(CreateBuiltinEndpoint(endpoint, endpointHandle));
    HcommResMgr::RegisterDeviceResetCallback();
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
{
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginEndpointDestroy(endpointHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    s32 devLogicIdSigned = HcclGetThreadDeviceId();
    CHK_PRT_RET(devLogicIdSigned < 0,
        HCCL_ERROR("[%s] HcclGetThreadDeviceId failed, ret[%d]", __func__, devLogicIdSigned), HCCL_E_INTERNAL);
    EndpointMonitor::GetInstance(devLogicIdSigned).RemoveEpHandleFromEndpointMonitor(endpointHandle);
    auto ret = g_EndpointMap.RemoveEndpoint(endpointHandle);
    CHK_PRT_RET(ret == false, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    endpointHandle = nullptr;

    return HCCL_SUCCESS;
}


HcommResult HcommEndpointStartListen(EndpointHandle endpointHandle, uint32_t port, HcommEndpointListenConfig* config)
{
    (void)config;
    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointStopListen(EndpointHandle endpointHandle, uint32_t port)
{
    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->ServerSocketStopListen(port));
    return HCCL_SUCCESS;
}

HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t *port)
{
    CHK_PTR_NULL(port);
    (void)HcommResMgrInit();
#ifdef ENABLE_EXPERIMENTAL
    if (IsPluginEndpoint(endpointHandle)) {
        return HCCL_E_NOT_SUPPORT;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return endpoint->ServerSocketGetListenPort(port);
}

HcommResult HcommMemReg(EndpointHandle endpointHandle, const char *memTag, const CommMem *mem,
    HcommMemHandle *memHandle)
{
    CHK_PTR_NULL(memHandle);
    EXCEPTION_HANDLE_BEGIN
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemReg(endpointHandle, memTag, mem, memHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->RegisterMemory(*mem, memTag, reinterpret_cast<void **>(memHandle)));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    CHK_PTR_NULL(memHandle);
    (void)HcommResMgrInit();
    EXCEPTION_HANDLE_BEGIN
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemUnreg(endpointHandle, memHandle, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->UnregisterMemory(memHandle));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void **memDesc,
    uint32_t *memDescLen)
{
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemExport(endpointHandle, memHandle, memDesc, memDescLen, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryExport(memHandle, memDesc, memDescLen));
    return HCCL_SUCCESS;
}

HcommResult HcommMemImport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen, CommMem *outMem)
{
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);
    CHK_PRT_RET(descLen == 0, HCCL_ERROR("[%s] descLen[0] is invalid", __func__), HCCL_E_PARA);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemImport(endpointHandle, memDesc, descLen, outMem, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_PTR_NULL(outMem);
    CommMem importedMem{};
    CHK_RET(endpoint->MemoryImport(memDesc, descLen, &importedMem));
    *outMem = importedMem;
    return HCCL_SUCCESS;
}

HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen)
{
    CHK_PTR_NULL(memDesc);
    (void)HcommResMgrInit();
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginMemUnimport(endpointHandle, memDesc, descLen, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(endpoint->MemoryUnimport(memDesc, descLen));
    return HCCL_SUCCESS;
}

/* 暂未实现 */
HcommResult HcommMemGrant(EndpointHandle endpointHandle, const HcommMemGrantInfo *remoteGrantInfo)
{
    CHK_PTR_NULL(remoteGrantInfo);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].",__func__, endpointHandle);

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->MemoryGrant(remoteGrantInfo));
    return HCCL_SUCCESS;
}

/* 暂未实现 */
HcommResult HcommMemRemap(const EndpointHandle endpointHandle, const CommMem *memArray, uint64_t arraySize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommMemGetAllMemHandles(EndpointHandle endpointHandle, void **memHandles, uint32_t *memHandleNum)
{
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);

    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    CHK_PRT_RET(endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]",
        __func__, endpointHandle), HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->GetAllMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

// 集合通信使用，待归一到HcommChannelCreate
HcommResult HcommCollectiveChannelCreate(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]",
        __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].",
        __func__, endpointHandle, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);

    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));
    return ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
}

HcommResult HcommChannelUpdateMemInfo(HcommMemHandle *memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
{
    CHK_PTR_NULL(memHandles);
    CHK_PRT_RET((memHandleNum == 0), HCCL_ERROR("[%s]Invalid memHandleNum, memHandleNum is 0.", __func__),
        HCCL_E_PARA);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelUpdateMemInfo(channelHandle, memHandles, memHandleNum, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    return ChannelProcess::ChannelUpdateMemInfo(memHandles, memHandleNum, channelHandle);
}

HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine,
    HcommChannelDesc *channelDescs, uint32_t channelNum, ChannelHandle *channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]",
        __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].",
        __func__, endpointHandle, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);
    auto endpoint = g_EndpointMap.GetEndpoint(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    (void)HcommResMgrInit();
    CHK_RET(ValidateCommEngineAgainstHostMode(engine));
    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(NormalizeHcommChannelDescs(channelDescs, channelNum, channelDescFinals)));

#ifdef ENABLE_EXPERIMENTAL
    bool pluginHandled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelCreate(endpointHandle, engine,
        channelDescFinals.data(), channelNum, channels, pluginHandled)));
    if (pluginHandled) {
        return HCCL_SUCCESS;
    }
#endif

    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle* targetChannels = hostChannelHandles.data();

    CHK_RET(ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescFinals.data(), channelNum,
        targetChannels));
    CHK_RET(ChannelProcess::ConnectChannels(targetChannels, channelNum, engine));
    CHK_RET(EnsureKernelBinLoaded(engine));
    // INNOTODO: 这里为什么使用的是Normalize前的channelDescs
    CHK_RET(ChannelProcess::SaveChannels(targetChannels, channels, channelDescFinals.data(), channelNum, engine, g_BinHandle));

    return HCCL_SUCCESS;
}

HcommResult HcommChannelGet(ChannelHandle channelHandle, void **channel)
{
    CHK_PTR_NULL(channel);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGet(channelHandle, channel, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif
    return ChannelProcess::ChannelGet(channelHandle, channel);
}

HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum,  int32_t* statusList)
{
    // 当前为非阻塞式建链，直接返回成功
    // 参数校验
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(statusList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s]Invalid listNum, listNum[%u]",
        __func__, listNum), HCCL_E_PARA);
    (void)HcommResMgrInit();
    // 为每个通道设置成功状态
    for (uint32_t i = 0; i < listNum; i++) {
#ifdef ENABLE_EXPERIMENTAL
        bool handled = false;
        CHK_RET(static_cast<HcclResult>(PluginChannelGetStatus(channelList[i], &statusList[i], handled)));
        if (handled) {
            continue;
        }
#endif
        statusList[i] = 0;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum)
{
    CHK_PTR_NULL(notifyNum);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGetNotifyNum(channelHandle, notifyNum, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif
    return ChannelProcess::ChannelGetNotifyNum(channelHandle, notifyNum);
}

HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    (void)HcommResMgrInit();
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]",
        __func__, channelNum), HCCL_E_PARA);
    std::vector<ChannelHandle> builtinChannels;
    builtinChannels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
#ifdef ENABLE_EXPERIMENTAL
        bool handled = false;
        CHK_RET(static_cast<HcclResult>(PluginChannelDestroy(channels[idx], handled)));
        if (handled) {
            continue;
        }
#endif
        builtinChannels.push_back(channels[idx]);
    }
    if (builtinChannels.empty()) {
        return HCCL_SUCCESS;
    }
    return ChannelProcess::ChannelDestroy(builtinChannels.data(), builtinChannels.size(), g_BinHandle);
}

HcommResult HcommChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t *memNum, CommMem **remoteMem, char ***memInfos)
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    CHK_PTR_NULL(memInfos);
#ifdef ENABLE_EXPERIMENTAL
    bool handled = false;
    CHK_RET(static_cast<HcclResult>(PluginChannelGetRemoteMems(channelHandle, memNum, remoteMem, memInfos, handled)));
    if (handled) {
        return HCCL_SUCCESS;
    }
#endif

    return ChannelProcess::ChannelGetRemoteMems(channelHandle, memNum, remoteMem, memInfos);
}

HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread,
    ThreadHandle *threads) {
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(notifyNumPerThread);
    (void)HcommResMgrInit();
    const uint32_t notifyNum = notifyNumPerThread[0];
    if (threadNum > 1U) {
        HCCL_RUN_WARNING("[%s] only notifyNumPerThread[0] is used currently, threadNum[%u], notifyNum[0][%u].",
            __func__, threadNum, notifyNum);
    }
    HCCL_INFO("[%s] ThreadAcquire begin. engine[%s], threadNum[%u], notifyPerThread[%u], threads[%p]",
        __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum, threads);
    CHK_RET(RefreshCommEngineContext(engine));

    // 1. 参数校验
    CHK_RET(hccl::ValidateThreadParams(threadNum, notifyNum));

    // 2. 获取引擎对应的类型
    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(hccl::CommEngineToStreamType(engine, streamType));

    // 3. 创建线程
    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    hccl::ThreadCreateParams params(engine, threadNum, notifyNum, notifyLoadType, streamType);
    CHK_RET(hccl::CreateAndInitThreads(params, newThreads));

    // 4. 插入全局映射表
    CHK_RET(hccl::SaveThreads(newThreads));

    // 5. 储存线程句柄
    CHK_RET(EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, g_BinHandle));

    HCCL_INFO("[HcommThreadAlloc] ThreadAcquire done: engine[%s] threadNum[%u], notifyPerThread[%u]",
              GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread,
    ThreadHandle *threads)
{
    return ::HcommThreadAlloc(engine, threadNum, &notifyNumPerThread, threads);
}

HcommResult HcommThreadAllocWithConfig(CommEngine engine, uint32_t threadNum,
    ThreadType type, const ThreadConfig *config, ThreadHandle *threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    CHK_PRT_RET(type == THREAD_TYPE_INVALID, HCCL_ERROR("[%s] thread type[%d] is invalid",
        __func__, static_cast<int32_t>(type)), (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_CPU_TS,
        HCCL_ERROR("[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use engine with ThreadType instead",
        __func__, static_cast<int32_t>(engine)), (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(engine == COMM_ENGINE_AIV || engine == COMM_ENGINE_CCU,
        HCCL_ERROR("[%s] commEngine[%d] AIV/CCU not supported, supported engines: CPU/AICPU",
        __func__, static_cast<int32_t>(engine)), (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(threadNum == 0,
        HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), (HcommResult)HCCL_E_PARA);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(engine));

    HCCL_INFO("[%s] begin. engine[%d], threadType[%d], threadNum[%u], threads[%p]",
        __func__, engine, static_cast<int32_t>(type), threadNum, threads);

    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(hccl::GetStreamType(engine, type, streamType));

    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    newThreads.reserve(threadNum);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR("[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first",
                __func__, i, config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD), (HcommResult)HCCL_E_PARA);
        CHK_RET(hccl::ValidateThreadParams(1, config[i].notifyNumPerThread));
        std::shared_ptr<hccl::Thread> threadPtr;
        HcclResult ret = hccl::CreateThread(engine, streamType, config[i].notifyNumPerThread, notifyLoadType, threadPtr);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] Failed to create thread at index[%u], ret[%d]", __func__, i, ret), (HcommResult)ret);
        ret = threadPtr->Init();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] Failed to init thread at index[%u], ret[%d]", __func__, i, ret), (HcommResult)ret);
        newThreads.emplace_back(std::move(threadPtr));
    }

    CHK_RET(hccl::SaveThreads(newThreads));
    CHK_RET(EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, g_BinHandle));

    HCCL_INFO("[%s] done: engine[%d] threadType[%d] threadNum[%u]",
        __func__, engine, static_cast<int32_t>(type), threadNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadFree(const ThreadHandle *threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    (void)HcommResMgrInit();
    return hccl::FreeThreads(threads, threadNum, g_BinHandle);
}

HcommResult HcommThreadAllocWithStream(CommEngine engine,
    rtStream_t stream, uint32_t notifyNum, ThreadHandle *thread)
{
    CHK_PTR_NULL(thread);
    hccl::NotifyLoadType notifyLoadType;
    CHK_RET(CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<hccl::Thread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<hccl::CpuTsThread>(stream, notifyNum, notifyLoadType), return HCCL_E_PTR);
    CHK_RET(handle->Init());
 
    // 返回第一个句柄
    *thread = reinterpret_cast<ThreadHandle>(handle.get());
    hcomm::g_ThreadMap.emplace(*thread , handle);
 
    HCCL_INFO("[ThreadMgr]  ThreadAcquireWithStream done: engine[%s] stream[%p],"
        "notifyNum[%u]", GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommEngineCtxCreate(CommEngine engine, uint64_t size, void **ctx)
{
    CHK_PTR_NULL(ctx);
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS
        || engine == COMM_ENGINE_CCU) {
        *ctx = malloc(size);
        CHK_PTR_NULL(*ctx);
        auto ret = memset_s(*ctx, size, 0, size);
        if (ret != EOK) {
            HCCL_ERROR("[%s] memset_s failed, ret[%d]", __func__, ret);
            free(*ctx);
            *ctx = nullptr;
            return HCCL_E_INTERNAL;
        }
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS
        || engine == COMM_ENGINE_AIV) {
        CHK_RET(hrtMalloc(ctx, size));
    } else {
        HCCL_ERROR("[%s] not support engine type[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommEngineCtxDestroy(CommEngine engine, void *ctx)
{
    CHK_PTR_NULL(ctx);
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS
        || engine == COMM_ENGINE_CCU) {
        free(ctx);
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS
        || engine == COMM_ENGINE_AIV) {
        CHK_RET(hrtFree(ctx));
    } else {
        HCCL_ERROR("[%s] invalid engine[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommEngineCtxCopy(CommEngine engine, void *dstCtx, const void *srcCtx, uint64_t size)
{
    CHK_PTR_NULL(dstCtx);
    CHK_PTR_NULL(srcCtx);
    if (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU
        || engine == COMM_ENGINE_AIV) {
        // 从Host内存拷贝到Device Context内存上
        CHK_RET(hrtMemSyncCopy(reinterpret_cast<uint8_t*>(dstCtx), size, srcCtx, size,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    } else if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS
        || engine == COMM_ENGINE_CCU) {
        CHK_SAFETY_FUNC_RET(memcpy_s(reinterpret_cast<uint8_t*>(dstCtx), size, srcCtx, size));
    } else {
        HCCL_ERROR("[%s]copy engine ctx failed, Unsupported engine[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO("[%s]copy engine ctx success, engine[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcommResult HcommDfxKernelLaunch(const std::string &commTag, aclrtBinHandle binHandle, HcclDfxOpInfo dfxOpInfo)
{
    // 申请device侧内存
    hccl::DeviceMem devicePackBuf = hccl::DeviceMem::alloc(sizeof(dfxOpInfo));
    CHK_PTR_NULL(devicePackBuf.ptr());
    
    // 将dfxOpInfo信息传递给device侧
    CHK_RET(hrtMemSyncCopy(devicePackBuf.ptr(),
        sizeof(dfxOpInfo),
        &dfxOpInfo,
        sizeof(dfxOpInfo),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 创建局部流
    hccl::Stream localStream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    // 下kernel
    std::string kernelName = "RunAicpuDfxOpInfoInitV2";

    struct InitTask {
        u64 context;
        char commTag[256];
    };

    InitTask customInitTask = {0, ""};
    customInitTask.context = reinterpret_cast<u64>(devicePackBuf.ptr());
    s32 sRet = strncpy_s(customInitTask.commTag, TAG_MAX_LENGTH, commTag.c_str(), TAG_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);

    CHK_RET(hccl::AicpuAclKernelLaunch(localStream.ptr(),
        reinterpret_cast<void *>(&customInitTask),  
        sizeof(customInitTask),  
        binHandle,            
        kernelName,
        true,
        NOTIFY_DEFAULT_WAIT_TIME));

    CHK_RET(
        hcclStreamSynchronize(localStream.ptr(), hccl::CommConfiger::GetInstance().GetCommConfigExecTimeOut(commTag)));

    HCCL_INFO("[%s] channel kernel launch success.", __func__);

    return HCCL_SUCCESS;
}

HcommResult HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc *endpointDesc, bool *value)
{
    CHK_PTR_NULL(endpointDesc);
    CHK_PTR_NULL(value);
    (void)HcommResMgrInit();

    return static_cast<HcommResult>(Endpoint::CheckFeature(*endpointDesc, featureType, *value));
}
