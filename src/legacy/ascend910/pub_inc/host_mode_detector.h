/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOST_MODE_DETECTOR_H
#define HOST_MODE_DETECTOR_H

#include "hccl_types.h"

namespace hccl {
/**
 * @brief Host 运行模式判定器。
 *
 * 引入背景：HCOMM 既要支持 NPU 服务器，也要支持无 NPU 的通用服务器（host-only)。
 * 历史代码里"NPU 调用失败"和"通用服务器无 NPU"共用同一条错误处理路径，
 * 会导致 NPU 场景下 fail-fast 失效或 host-only 场景下意外走进 NPU 路径。
 *
 * 本判定器把"当前进程是否处在 host-only 模式"显式化为全局状态：
 *   1) 优先读环境变量 `HCOMM_HOST_ONLY`：`1` 强制 HostOnly，`0` 强制 Device；
 *   2) 否则由 `ResolveRuntimeDevicePhyId`（base_comm 层）在第一次探测到
 *      "无 NPU 设备"时回填 `SetFromProbe(true)`；显式有 NPU 时回填 `false`；
 *   3) 未被任何一方触发时按 `Unknown` 处理，适配层不进入 HostOnly 兜底分支，
 *      行为与历史代码完全一致。
 *
 * 所有 NPU 运行时适配函数（hrtGetDevice / hrtGetDeviceType / HrtGetDevice /
 * HrtGetSocVer / GetRunSideIsDevice / ...）只在 `IsHostOnly() == true` 时
 * 启用 host 兜底分支，NPU 分支保持字节级不变；调用点因此无需新增 try/catch。
 */
enum class HostMode {
    HOST_MODE_UNKNOWN = 0,  // 未探测，所有适配函数保持原有 fail-fast 行为
    HOST_MODE_HOST_ONLY,     // 通用服务器/无 NPU，启用 host 兜底
    HOST_MODE_DEVICE,       // NPU 模式，所有调用按原语义执行
};

class HostModeDetector {
public:
    /**
     * @brief 取当前 Host 模式。线程安全、幂等。
     *        优先级：env HCOMM_HOST_ONLY > 探测回填。
     */
    static HostMode GetMode();

    /** @brief 便捷谓词：等价于 GetMode() == HOST_MODE_HOST_ONLY。 */
    static bool IsHostOnly();

    /**
     * @brief 由 base_comm 层的 NPU 探测路径把"是否无 NPU"回填进来。
     *        只在 env 未显式指定时生效；可被多次调用，以第一次非 UNKNOWN 为准。
     */
    static void SetFromProbe(bool noDevice);
};

} // namespace hccl

#endif // HOST_MODE_DETECTOR_H