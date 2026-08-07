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

#include "relu_grad_v4.h"

// ReluGradV4(gradients<T>, mask<uint8>) -> backprops<T> = mask ? gradients : 0.
// The 2nd GM address is the uint8 mask (parameter kept named "features" for prototype compat).
template <typename D_T_X>
__global__ __aicore__ void relu_grad_v4(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace,
                                        GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluGradV4TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluGradV4TilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (std::is_same_v<D_T_X, float> || std::is_same_v<D_T_X, half>) {
        NsReluGradV4::KernelReluGradSelect<D_T_X> op; // native Select in T (grad not cast)
        op.Init(gradients, features, backprops, &tilingData, &pipe);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, bfloat16_t>) {
        NsReluGradV4::KernelReluGradCastSelect<D_T_X> op; // bf16 -> fp32 Select -> bf16
        op.Init(gradients, features, backprops, &tilingData, &pipe);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, int32_t>) {
        NsReluGradV4::KernelReluGradInt32<D_T_X> op; // grad * mask{0,1} (int32 mul, exact)
        op.Init(gradients, features, backprops, &tilingData, &pipe);
        op.Process();
    } else {                                        // int8_t / uint8_t
        NsReluGradV4::KernelReluGradInt8<D_T_X> op; // cast grad->half, half Select, cast back
        op.Init(gradients, features, backprops, &tilingData, &pipe);
        op.Process();
    }
}
