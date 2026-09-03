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
 * =============================================================================
 * optim/inplace_apply_proximal_gradient_descent/op_kernel/arch35/inplace_apply_proximal_gradient_descent.cpp
 * =============================================================================
 * Role: DESIGN §10.1 Kernel 入口。dtype 来自 OpDef profile 注入的 DTYPE_VAR
 *       编译宏；唯一模板参数 BUFFER_MODE 静态决定 Queue 深度 1/2。入口为 AIV-only；
 *       var_out 是 aliasing:none 的真实非 inplace Output；workspace 保留 ABI
 *       位置但不解引用；use_locking 是图属性而不是 Kernel 参数。
 * =============================================================================
 */

#include "kernel_operator.h"
#include "inplace_apply_proximal_gradient_descent_tiling_key.h"
#include "inplace_apply_proximal_gradient_descent_tiling_data.h"
#include "inplace_apply_proximal_gradient_descent.h"

template <int BUFFER_MODE>
__global__ __aicore__ void inplace_apply_proximal_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2,
                                                                   GM_ADDR delta, GM_ADDR var_out, GM_ADDR workspace,
                                                                   GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(InplaceApplyProximalGradientDescentTilingData);
    GET_TILING_DATA_WITH_STRUCT(InplaceApplyProximalGradientDescentTilingData, tilingData, tiling);

    (void)workspace;
    NsInplaceApplyProximalGradientDescent::Kernel<DTYPE_VAR, BUFFER_MODE> op;
    op.Init(var, alpha, l1, l2, delta, var_out, &tilingData);
    op.Process();
}
