/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_apply_adagrad.cpp
 * \brief InplaceApplyAdagrad arch35 kernel entry.
 */
#include "inplace_apply_adagrad_tiling_key.h"
#include "inplace_apply_adagrad.h"

using namespace AscendC;
using namespace InplaceApplyAdagradTilingData;

template <uint64_t schMode, uint64_t updateSlots, uint64_t dType>
__global__ __aicore__ void inplace_apply_adagrad(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR var_out,
                                                 GM_ADDR accum_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(InplaceApplyAdagradTilingDataStruct);
    GET_TILING_DATA_WITH_STRUCT(InplaceApplyAdagradTilingDataStruct, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (dType == INPLACE_APPLY_ADAGRAD_TPL_FP16) {
        NsInplaceApplyAdagrad::InplaceApplyAdagradKernel<half, (updateSlots > 0)> op;
        op.Init(var, accum, lr, grad, var_out, accum_out, &tilingData);
        op.Process();
    } else if constexpr (dType == INPLACE_APPLY_ADAGRAD_TPL_BF16) {
        NsInplaceApplyAdagrad::InplaceApplyAdagradKernel<bfloat16_t, (updateSlots > 0)> op;
        op.Init(var, accum, lr, grad, var_out, accum_out, &tilingData);
        op.Process();
    } else {
        NsInplaceApplyAdagrad::InplaceApplyAdagradKernel<float, (updateSlots > 0)> op;
        op.Init(var, accum, lr, grad, var_out, accum_out, &tilingData);
        op.Process();
    }
}
