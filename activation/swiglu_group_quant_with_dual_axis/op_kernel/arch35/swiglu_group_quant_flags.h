/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_GROUP_QUANT_FLAGS_H
#define SWIGLU_GROUP_QUANT_FLAGS_H
#include <cstdint>

// Shared Host/Kernel ABI for the single-axis mode 5 and dual-axis MX paths.
enum SwigluGroupQuantMxFlag : uint32_t {
    MX_HAS_WEIGHT = 1U << 0,
    MX_HAS_GROUP = 1U << 1,
    MX_HAS_CLAMP = 1U << 2,
    MX_OUTPUT_ORIGIN = 1U << 3,
    MX_SHARE_ORIGIN = 1U << 4,
    MX_OWN_WHOLE_GROUPS = 1U << 5,
};
#endif
