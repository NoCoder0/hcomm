/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_mode_detector.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "log.h"

namespace hccl {
namespace {
constexpr const char *HOST_ONLY_ENV = "HCOMM_HOST_ONLY";

std::mutex g_mutex;
HostMode g_finalMode = HostMode::HOST_MODE_UNKNOWN;
bool g_decided = false;

HostMode ResolveFromEnv()
{
    const char *val = std::getenv(HOST_ONLY_ENV);
    if (val == nullptr) {
        return HostMode::HOST_MODE_UNKNOWN;
    }
    if (std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0 || std::strcmp(val, "TRUE") == 0) {
        HCCL_INFO("[HostModeDetector] env %s=%s -> HOST_ONLY.", HOST_ONLY_ENV, val);
        return HostMode::HOST_MODE_HOST_ONLY;
    }
    if (std::strcmp(val, "0") == 0 || std::strcmp(val, "false") == 0 || std::strcmp(val, "FALSE") == 0) {
        HCCL_INFO("[HostModeDetector] env %s=%s -> DEVICE.", HOST_ONLY_ENV, val);
        return HostMode::HOST_MODE_DEVICE;
    }
    HCCL_WARNING("[HostModeDetector] env %s has invalid value '%s', ignored.", HOST_ONLY_ENV, val);
    return HostMode::HOST_MODE_UNKNOWN;
}

// 用 env 或探测结果第一次确定最终模式；env 优先，且一旦确定不再变更。
// probeHint 可为 HOST_MODE_UNKNOWN，表示仅查询当前状态；env 已确定时忽略 probe。
HostMode Decide(HostMode probeHint)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_decided) {
        return g_finalMode;
    }
    HostMode env = ResolveFromEnv();
    if (env != HostMode::HOST_MODE_UNKNOWN) {
        g_finalMode = env;
        g_decided = true;
        HCCL_INFO("[HostModeDetector] mode pinned to [%d] by env.", static_cast<int>(g_finalMode));
        return g_finalMode;
    }
    if (probeHint != HostMode::HOST_MODE_UNKNOWN) {
        g_finalMode = (probeHint == HostMode::HOST_MODE_HOST_ONLY) ? HostMode::HOST_MODE_HOST_ONLY
                                                                   : HostMode::HOST_MODE_DEVICE;
        g_decided = true;
        HCCL_INFO("[HostModeDetector] mode pinned to [%d] by probe.", static_cast<int>(g_finalMode));
        return g_finalMode;
    }
    return HostMode::HOST_MODE_UNKNOWN;
}
} // namespace

HostMode HostModeDetector::GetMode()
{
    return Decide(HostMode::HOST_MODE_UNKNOWN);
}

bool HostModeDetector::IsHostOnly()
{
    return GetMode() == HostMode::HOST_MODE_HOST_ONLY;
}

void HostModeDetector::SetFromProbe(bool noDevice)
{
    Decide(noDevice ? HostMode::HOST_MODE_HOST_ONLY : HostMode::HOST_MODE_DEVICE);
}
} // namespace hccl