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
 * \file ada_layer_norm_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "ada_layer_norm_grad_common.h"
#include "ada_layer_norm_grad_workspace.h"

using namespace AdaLayerNormGrad;

#define WORKSPACE_FLOAT_FLOAT 201
#define WORKSPACE_HALF_HALF 202
#define WORKSPACE_HALF_FLOAT 203
#define WORKSPACE_BF16_BF16 204
#define WORKSPACE_BF16_FLOAT 205
#define WORKSPACE_FLOAT_FLOAT_DETERMINISTIC 211
#define WORKSPACE_HALF_HALF_DETERMINISTIC 212
#define WORKSPACE_HALF_FLOAT_DETERMINISTIC 213
#define WORKSPACE_BF16_BF16_DETERMINISTIC 214
#define WORKSPACE_BF16_FLOAT_DETERMINISTIC 215

#define COMMON_FLOAT_FLOAT 401
#define COMMON_HALF_HALF 402
#define COMMON_HALF_FLOAT 403
#define COMMON_BFLOAT16_BFLOAT16 404
#define COMMON_BFLOAT16_FLOAT 405
#define COMMON_FLOAT_FLOAT_DETERMINISTIC 411
#define COMMON_HALF_HALF_DETERMINISTIC 412
#define COMMON_HALF_FLOAT_DETERMINISTIC 413
#define COMMON_BFLOAT16_BFLOAT16_DETERMINISTIC 414
#define COMMON_BFLOAT16_FLOAT_DETERMINISTIC 415

#define INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(Tdy, Tgamma, Isdeterministic)                 \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(AdaLayerNormGradTilingDataWorkspace, tilingData, tiling);   \
        AdaLayerNormGradWorkspace<Tdy, Tgamma, Isdeterministic> op;                             \
        op.Init(dy, x, rstd, mean, scale, gamma, beta, pd_x, pd_scale, pd_shift, pd_gamma, pd_beta, usrWorkspace, &tilingData, pipe); \
        op.Process(&tilingData);                                                               \
    } while (0)

#define INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(Tdy, Tgamma, Isdeterministic)                    \
    do {                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(AdaLayerNormGradTilingDataCommon, tilingData, tiling);      \
        AdaLayerNormGradCommon<Tdy, Tgamma, Isdeterministic> op;                                \
        op.Init(dy, x, rstd, mean, scale, gamma, beta, pd_x, pd_scale, pd_shift, pd_gamma, pd_beta, usrWorkspace, &tilingData, pipe); \
        op.Process(&tilingData);                                                               \
    } while (0)

extern "C" __global__ __aicore__ void ada_layer_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pd_x, GM_ADDR pd_scale, GM_ADDR pd_shift,
    GM_ADDR pd_gamma, GM_ADDR pd_beta, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (g_coreType == AIC) {
        return;
    }
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
 
    if (TILING_KEY_IS(COMMON_FLOAT_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_HALF)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_BFLOAT16)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(COMMON_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_HALF_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(COMMON_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_BFLOAT16_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(COMMON_BFLOAT16_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_COMMON_IMPL(bfloat16_t, float, true);
        return;
    }else if (TILING_KEY_IS(WORKSPACE_FLOAT_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(float, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_HALF)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(half, half, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(half, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_BF16)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(bfloat16_t, bfloat16_t, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_FLOAT)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(bfloat16_t, float, false);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_FLOAT_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(float, float, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_HALF_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(half, half, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_HALF_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(half, float, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_BF16_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(bfloat16_t, bfloat16_t, true);
        return;
    } else if (TILING_KEY_IS(WORKSPACE_BF16_FLOAT_DETERMINISTIC)) {
        INVOKE_ADA_LAYER_NORM_GRAD_WORKSPACE_IMPL(bfloat16_t, float, true);
        return;
    }

    return;
}