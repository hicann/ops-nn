/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_NPU_ARCH_UTIL_H_
#define OP_HOST_NPU_ARCH_UTIL_H_

#include <string>
#include "log/log.h"
#include "runtime/runtime/base.h"

namespace ops {

/**
 * @brief Check if current platform is DAV_3510 arch (covers both Ascend950 and Ascend350).
 *
 * @return true if current platform is DAV_3510, false otherwise or on failure.
 */
inline bool IsDav3510Arch()
{
    constexpr uint32_t kMaxLen = 32;
    char npuArchStr[kMaxLen] = {};
    if (rtGetSocSpec("version", "NpuArch", npuArchStr, kMaxLen) != 0) {
        OP_LOGW("IsDav3510Arch", "rtGetSocSpec failed, cannot get NpuArch.");
        return false;
    }
    OP_LOGD("IsDav3510Arch", "Current NpuArch: %s", npuArchStr);
    return std::string(npuArchStr) == "3510";
}

} // namespace ops

#endif // OP_HOST_NPU_ARCH_UTIL_H_
