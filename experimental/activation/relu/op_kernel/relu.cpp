/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "relu.h"

template <typename D_T_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);

    if constexpr (std::is_same_v<D_T_X, float> || std::is_same_v<D_T_X, half> || std::is_same_v<D_T_X, int32_t>) {
        NsRelu::KernelRelu<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, bfloat16_t>) {
        NsRelu::KernelReluUpcast<D_T_X, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, int8_t>) {
        NsRelu::KernelReluUpcast<D_T_X, half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        NsRelu::KernelReluScalarInt64<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
