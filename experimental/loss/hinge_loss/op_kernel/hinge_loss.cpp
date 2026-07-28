/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hinge_loss.h"

#ifdef __CCE_KT_TEST__
template <uint32_t schMode>
#else
extern "C"
#endif
__global__ __aicore__ void hinge_loss(GM_ADDR predict, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HingeLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(HingeLossTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

#ifdef __CCE_KT_TEST__
    // Kernel UT instantiates all supported dtypes in one test binary and uses schMode to select the concrete type.
    // Production builds provide the concrete type through DTYPE_PREDICT.
    if constexpr (schMode == 0) {
        NsHingeLoss::HingeLoss<float> op;
        op.Init(predict, target, loss, &tilingData, pipe);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsHingeLoss::HingeLoss<half> op;
        op.Init(predict, target, loss, &tilingData, pipe);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsHingeLoss::HingeLoss<bfloat16_t> op;
        op.Init(predict, target, loss, &tilingData, pipe);
        op.Process();
    }
#else
    NsHingeLoss::HingeLoss<DTYPE_PREDICT> op;
    op.Init(predict, target, loss, &tilingData, pipe);
    op.Process();
#endif
}
