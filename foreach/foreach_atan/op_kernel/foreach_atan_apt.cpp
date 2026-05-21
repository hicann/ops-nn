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
 * \file foreach_atan_apt.cpp
 * \brief kernel entry for foreach_atan
 */

#include "arch35/foreach_atan_simt.h"

enum class ForeachAtanTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void foreach_atan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachAtanTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachAtanTilingData, tilingData, tiling);

    // Access tiling data via GM pointer to avoid stack overflow with large struct
    const __gm__ ForeachAtanTilingData* tilingGm =
        reinterpret_cast<const __gm__ ForeachAtanTilingData*>(tiling);

    if constexpr (schMode == static_cast<uint32_t>(ForeachAtanTilingKey::TILING_KEY_FLOAT)) {
        NsForeachAtan::Process<float>(x, y, tilingGm);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachAtanTilingKey::TILING_KEY_FLOAT16)) {
        NsForeachAtan::Process<half>(x, y, tilingGm);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachAtanTilingKey::TILING_KEY_BF16)) {
        NsForeachAtan::Process<bfloat16_t>(x, y, tilingGm);
    }
}