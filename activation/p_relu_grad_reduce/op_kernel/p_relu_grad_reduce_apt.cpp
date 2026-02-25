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
 * \file p_relu_grad_reduce_apt.cpp
 * \brief prelu_grad_reduce
 */

#include "atvoss/reduce/reduce_sch.h"
#include "arch35/prelu_grad_reduce_dag.h"
#include "arch35/prelu_grad_reduce_struct.h"

using namespace Ops::Base::ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void prelu_grad_reduce(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
                                             GM_ADDR updates, GM_ADDR da, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(ReduceOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceOpTilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_GRADS>::T;
    using Op = ReduceSch<REDUCE_TPL_VALUE, PreluGradReduce::PreluGradReduceDag<DTYPE_GRADS, PromoteType>::OpDag>;
    Op op(&tilingData);
    op.Init(&pipe, updates, da, userWS);
    op.Process();
}