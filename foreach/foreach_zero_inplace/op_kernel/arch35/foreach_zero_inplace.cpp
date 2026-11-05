/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "foreach_zero_inplace_simd.h"

enum class ForeachZeroInplaceTilingKey : uint32_t {
    FLOAT16 = FOREACH_ZERO_INPLACE_SCH_MODE_FLOAT16,
    FLOAT = FOREACH_ZERO_INPLACE_SCH_MODE_FLOAT,
    INT32 = FOREACH_ZERO_INPLACE_SCH_MODE_INT32,
    BF16 = FOREACH_ZERO_INPLACE_SCH_MODE_BF16,
    INT16 = FOREACH_ZERO_INPLACE_SCH_MODE_INT16,
};

template <uint32_t schMode>
__global__ __aicore__ void foreach_zero_inplace(GM_ADDR x, GM_ADDR xRef, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)x;
    REGISTER_TILING_DEFAULT(ForeachZeroInplaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachZeroInplaceTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(ForeachZeroInplaceTilingKey::FLOAT16)) {
        ForeachZeroInplaceA5::Process<half>(xRef, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachZeroInplaceTilingKey::FLOAT)) {
        ForeachZeroInplaceA5::Process<float>(xRef, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachZeroInplaceTilingKey::INT32)) {
        ForeachZeroInplaceA5::Process<int32_t>(xRef, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachZeroInplaceTilingKey::BF16)) {
        ForeachZeroInplaceA5::Process<bfloat16_t>(xRef, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachZeroInplaceTilingKey::INT16)) {
        ForeachZeroInplaceA5::Process<int16_t>(xRef, &tilingData);
    }
}
