/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/**
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/**
 * \file hard_swish_arch35.cpp
 * \brief HardSwish 算子 kernel 入口（arch35 - Ascend950）
 *
 * 模板分发：
 *   schMode=0 (FP32): HardSwish<float>
 *   schMode=1 (FP16): HardSwishFp16（提升至 fp32 计算）
 *   schMode=2 (BF16): HardSwishBf16（提升至 fp32 计算）
 */

#include "arch35/hard_swish.h"

enum class HardSwishTilingKey : uint32_t {
    TILING_KEY_FP32 = 0,
    TILING_KEY_FP16 = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void hard_swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSwishTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSwishTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_FP32)) {
        NsHardSwish::HardSwish<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_FP16)) {
        NsHardSwish::HardSwishFp16 op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(HardSwishTilingKey::TILING_KEY_BF16)) {
        NsHardSwish::HardSwishBf16 op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
