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
 * \file instance_norm_grad.cpp
 * \brief InstanceNormGrad arch35 (Ascend950 / regbase) kernel entry.
 *
 * TilingKey: full_load 101(fp32)/102(fp16), recompute 301(fp32)/302(fp16), empty 500.
 * Input order follows A2: (dy, x, variance, mean, gamma) -> (pd_x, pd_gamma, pd_beta). No attrs.
 */
#include "kernel_operator.h"
#include "arch35/instance_norm_grad_full_load.h"
#include "arch35/instance_norm_grad_recompute.h"
#include "arch35/instance_norm_grad_empty_load.h"

using namespace InstanceNormGrad;

namespace {
#define INSTANCE_NORM_GRAD_FULL_LOAD_FLOAT 101
#define INSTANCE_NORM_GRAD_FULL_LOAD_HALF 102
#define INSTANCE_NORM_GRAD_RECOMPUTE_FLOAT 301
#define INSTANCE_NORM_GRAD_RECOMPUTE_HALF 302
#define INSTANCE_NORM_GRAD_EMPTY_TENSOR_KEY 500
} // namespace

extern "C" __global__ __aicore__ void instance_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean,
                                                         GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
                                                         GM_ADDR workspace, GM_ADDR tilingdata)
{
    if (workspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(INSTANCE_NORM_GRAD_FULL_LOAD_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormGradTilingData, tilingIn, tilingdata);
        const InstanceNormGradTilingData* __restrict tiling = &tilingIn;
        TPipe pipe;
        InstanceNormGradFullLoad<float> op;
        op.Init(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(INSTANCE_NORM_GRAD_FULL_LOAD_HALF)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormGradTilingData, tilingIn, tilingdata);
        const InstanceNormGradTilingData* __restrict tiling = &tilingIn;
        TPipe pipe;
        InstanceNormGradFullLoad<half> op;
        op.Init(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(INSTANCE_NORM_GRAD_RECOMPUTE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormGradTilingData, tilingIn, tilingdata);
        const InstanceNormGradTilingData* __restrict tiling = &tilingIn;
        TPipe pipe;
        InstanceNormGradReCompute<float> op;
        op.Init(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(INSTANCE_NORM_GRAD_RECOMPUTE_HALF)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormGradTilingData, tilingIn, tilingdata);
        const InstanceNormGradTilingData* __restrict tiling = &tilingIn;
        TPipe pipe;
        InstanceNormGradReCompute<half> op;
        op.Init(dy, x, variance, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(INSTANCE_NORM_GRAD_EMPTY_TENSOR_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormGradEmptyTilingData, tilingIn, tilingdata);
        const InstanceNormGradEmptyTilingData* __restrict tiling = &tilingIn;
        TPipe pipe;
        EmptyDgamma<DTYPE_DY, 2> op(&pipe, tiling);
        op.Init(pd_gamma, pd_beta);
        op.Process();
    }
}
