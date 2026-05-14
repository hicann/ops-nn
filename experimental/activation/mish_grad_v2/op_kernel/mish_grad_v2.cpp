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

#include "mish_grad_v2.h"

template <typename D_T_X>
__global__ __aicore__ void mish_grad_v2(
    GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MishGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MishGradV2TilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (std::is_same_v<D_T_X, float>) {
        NsMishGradV2::KernelMishGradFloat op;
        op.Init(grad, x, tanhx, x_grad, &tilingData, &pipe);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, half>) {
        NsMishGradV2::KernelMishGradCast<D_T_X> op;
        op.Init(grad, x, tanhx, x_grad, &tilingData, &pipe);
        op.Process();
    } else {
        NsMishGradV2::KernelMishGradCast<D_T_X> op;
        op.Init(grad, x, tanhx, x_grad, &tilingData, &pipe);
        op.Process();
    }
}
