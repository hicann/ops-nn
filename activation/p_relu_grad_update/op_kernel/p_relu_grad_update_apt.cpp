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
 * \file p_relu_grad_update_apt.cpp
 * \brief calculating the backpropagation of prelu operation
 * prelu equivalent function:prelu(x)=max(0,input_features)+input_weights*min(0,input_features)
 * so prelu_grad output_backprops: output_backprops_dx=input_features>0?input_gradients:input_weight*input_gradients
 * output_backprops_da=input_features>0?0:input_features*input_gradients
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/prelu_grad_update_dag.h"
#include "arch35/prelu_grad_update_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void prelu_grad_update(
    GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx,
    GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = PReluGradUpdate::PReluGradUpdateDAG<DTYPE_GRADS>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(grads, features, weights, dx, update);
    return;
}