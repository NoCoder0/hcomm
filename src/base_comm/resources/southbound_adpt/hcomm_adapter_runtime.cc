/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_adapter_runtime.h"

#include <cstdlib>
#include <cstring>

#include "adapter_rts_common.h"

namespace hcomm {

constexpr uint32_t kDefaultResourceId = 0U;

bool IsHostNicPluginForceLoadEnabled()
{
    const char *forceLoad = std::getenv(HCOMM_FORCE_HOST_NIC_PLUGIN_ENV);
    return forceLoad != nullptr && std::strcmp(forceLoad, "1") == 0;
}

HcclResult ResolveRuntimeDevicePhyId(uint32_t &devicePhyId, bool &noDevice)
{
    if (IsHostNicPluginForceLoadEnabled()) {
        devicePhyId = kDefaultResourceId;
        noDevice = true;
        HCCL_WARNING("[HcommAdapterRuntime][%s] force Host-only resource id[%u], env[%s]=1, test only.",
            __func__, devicePhyId, HCOMM_FORCE_HOST_NIC_PLUGIN_ENV);
        return HCCL_SUCCESS;
    }

    uint32_t deviceCount = 0;
    HcclResult ret = hrtGetDeviceCount(&deviceCount);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcommAdapterRuntime][%s] get device count failed, ret[%d], use resource id[%u].",
            __func__, ret, kDefaultResourceId);
        devicePhyId = kDefaultResourceId;
        noDevice = true;
        return HCCL_SUCCESS;
    }

    if (deviceCount == 0) {
        devicePhyId = kDefaultResourceId;
        noDevice = true;
        return HCCL_SUCCESS;
    }

    int32_t devLogicId = 0;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId), devicePhyId));
    noDevice = false;
    HCCL_INFO("[HcommAdapterRuntime][%s] deviceCount[%u], devLogicId[%d], devicePhyId[%u].",
        __func__, deviceCount, devLogicId, devicePhyId);
    return HCCL_SUCCESS;
}

} // namespace hcomm
