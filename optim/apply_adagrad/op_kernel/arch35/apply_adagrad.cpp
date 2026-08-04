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
 * \file apply_adagrad.cpp
 * \brief ApplyAdagrad arch35 kernel entry.
 */
#include "apply_adagrad_tiling_key.h"
#include "apply_adagrad.h"

using namespace AscendC;
using namespace ApplyAdagradTilingData;

template <uint64_t schMode, uint64_t updateSlots, uint64_t dType>
__global__ __aicore__ void apply_adagrad(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR var_out,
                                         GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyAdagradTilingDataStruct);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdagradTilingDataStruct, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    NsApplyAdagrad::ApplyAdagradKernel<DTYPE_VAR, (updateSlots > 0)> op;
    op.Init(var, accum, lr, grad, var_out, &tilingData);
    op.Process();
}
