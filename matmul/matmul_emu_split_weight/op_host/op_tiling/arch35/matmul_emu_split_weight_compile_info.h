/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_HOST_MATMUL_EMU_SPLIT_WEIGHT_COMPILE_INFO_H__
#define __OP_HOST_MATMUL_EMU_SPLIT_WEIGHT_COMPILE_INFO_H__
#include <cstdint>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct MatmulEmuSplitWeightCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t l1Size{0UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t l0cSize{0UL};
    uint64_t ubSize{0UL};
};
} // namespace optiling
#endif // __OP_HOST_MATMUL_EMU_SPLIT_WEIGHT_COMPILE_INFO_H__
