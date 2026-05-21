/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_minimum_list_apt.cpp
 * \brief kernel entry for foreach_minimum_list
 */

#include "arch35/foreach_minimum_list_simt.h"

enum class ForeachMinimumListTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_BF16 = 2,
    TILING_KEY_INT32 = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void foreach_minimum_list(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachMinimumListTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachMinimumListTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(ForeachMinimumListTilingKey::TILING_KEY_FLOAT)) {
        NsForeachMinimumList::Process<float, 0>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachMinimumListTilingKey::TILING_KEY_FLOAT16)) {
        NsForeachMinimumList::Process<half, 1>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachMinimumListTilingKey::TILING_KEY_BF16)) {
        NsForeachMinimumList::Process<bfloat16_t, 2>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachMinimumListTilingKey::TILING_KEY_INT32)) {
        NsForeachMinimumList::Process<int32_t, 3>(x1, x2, y, &tilingData);
    }
}
