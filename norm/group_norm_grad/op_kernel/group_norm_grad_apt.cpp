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
 * \file group_norm_grad_apt.cpp
 * \brief
 */

#include "arch35/group_norm_grad_g_full_load.h"
#include "arch35/group_norm_grad_c_full_load.h"
#include "arch35/group_norm_grad_recompute.h"
#include "arch35/group_norm_grad_small_ng_c_full_load.h"
#include "arch35/group_norm_grad_empty_load.h"

using namespace GroupNormGrad;

namespace {
#define G_FULL_LOAD_FLOAT_FLOAT 101
#define G_FULL_LOAD_HALF_HALF 102
#define G_FULL_LOAD_BF16_BF16 103
#define G_FULL_LOAD_HALF_FLOAT 104
#define G_FULL_LOAD_BF16_FLOAT 105
#define C_FULL_LOAD_FLOAT_FLOAT 201
#define C_FULL_LOAD_HALF_HALF 202
#define C_FULL_LOAD_BF16_BF16 203
#define C_FULL_LOAD_HALF_FLOAT 204
#define C_FULL_LOAD_BF16_FLOAT 205
#define RECOMPUTE_FLOAT_FLOAT 301
#define RECOMPUTE_HALF_HALF 302
#define RECOMPUTE_BF16_BF16 303
#define RECOMPUTE_HALF_FLOAT 304
#define RECOMPUTE_BF16_FLOAT 305
#define SMALL_NG_C_FULL_LOAD_FLOAT_FLOAT 401
#define SMALL_NG_C_FULL_LOAD_HALF_HALF 402
#define SMALL_NG_C_FULL_LOAD_BF16_BF16 403
#define SMALL_NG_C_FULL_LOAD_HALF_FLOAT 404
#define SMALL_NG_C_FULL_LOAD_BF16_FLOAT 405
#define GROUP_NORM_GRAD_EMPTY_TENSOR_KEY 500
} // namespace

extern "C" __global__ __aicore__ void group_norm_grad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, GM_ADDR tilingdata)
{
    if (workspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(GroupNormGradRegBaseTilingData, tiling_data_in, tilingdata);
    const GroupNormGradRegBaseTilingData* __restrict tiling_data = &tiling_data_in;
    TPipe pipe;
    if (TILING_KEY_IS(SMALL_NG_C_FULL_LOAD_FLOAT_FLOAT)) {
        GroupNormGradSmallNGCFullLoad<float, float> opFP32Det;
        opFP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(SMALL_NG_C_FULL_LOAD_HALF_HALF)) {
        GroupNormGradSmallNGCFullLoad<half, half> opFP16Det;
        opFP16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(SMALL_NG_C_FULL_LOAD_BF16_BF16)) {
        GroupNormGradSmallNGCFullLoad<bfloat16_t, bfloat16_t> opBF16Det;
        opBF16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16Det.Process();
    } else if (TILING_KEY_IS(SMALL_NG_C_FULL_LOAD_HALF_FLOAT)) {
        GroupNormGradSmallNGCFullLoad<half, float> opF16FP32Det;
        opF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opF16FP32Det.Process();
    } else if (TILING_KEY_IS(SMALL_NG_C_FULL_LOAD_BF16_FLOAT)) {
        GroupNormGradSmallNGCFullLoad<bfloat16_t, float> opBF16FP32Det;
        opBF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16FP32Det.Process();
    } else if (TILING_KEY_IS(G_FULL_LOAD_FLOAT_FLOAT)) {
        GroupNormGradGFullLoad<float, float> opFP32Det;
        opFP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(G_FULL_LOAD_HALF_HALF)) {
        GroupNormGradGFullLoad<half, half> opFP16Det;
        opFP16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(G_FULL_LOAD_BF16_BF16)) {
        GroupNormGradGFullLoad<bfloat16_t, bfloat16_t> opBF16Det;
        opBF16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16Det.Process();
    } else if (TILING_KEY_IS(G_FULL_LOAD_HALF_FLOAT)) {
        GroupNormGradGFullLoad<half, float> opF16FP32Det;
        opF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opF16FP32Det.Process();
    } else if (TILING_KEY_IS(G_FULL_LOAD_BF16_FLOAT)) {
        GroupNormGradGFullLoad<bfloat16_t, float> opBF16FP32Det;
        opBF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16FP32Det.Process();
    } else if (TILING_KEY_IS(C_FULL_LOAD_FLOAT_FLOAT)) {
        GroupNormGradCFullLoad<float, float> opFP32Det;
        opFP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(C_FULL_LOAD_HALF_HALF)) {
        GroupNormGradCFullLoad<half, half> opFP16Det;
        opFP16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(C_FULL_LOAD_BF16_BF16)) {
        GroupNormGradCFullLoad<bfloat16_t, bfloat16_t> opBF16Det;
        opBF16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16Det.Process();
    } else if (TILING_KEY_IS(C_FULL_LOAD_HALF_FLOAT)) {
        GroupNormGradCFullLoad<half, float> opF16FP32Det;
        opF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opF16FP32Det.Process();
    } else if (TILING_KEY_IS(C_FULL_LOAD_BF16_FLOAT)) {
        GroupNormGradCFullLoad<bfloat16_t, float> opBF16FP32Det;
        opBF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16FP32Det.Process();
    } else if (TILING_KEY_IS(RECOMPUTE_FLOAT_FLOAT)) {
        GroupNormGradReCompute<float, float> opFP32Det;
        opFP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(RECOMPUTE_HALF_HALF)) {
        GroupNormGradReCompute<half, half> opFP16Det;
        opFP16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(RECOMPUTE_BF16_BF16)) {
        GroupNormGradReCompute<bfloat16_t, bfloat16_t> opBF16Det;
        opBF16Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16Det.Process();
    } else if (TILING_KEY_IS(RECOMPUTE_HALF_FLOAT)) {
        GroupNormGradReCompute<half, float> opF16FP32Det;
        opF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opF16FP32Det.Process();
    } else if (TILING_KEY_IS(RECOMPUTE_BF16_FLOAT)) {
        GroupNormGradReCompute<bfloat16_t, float> opBF16FP32Det;
        opBF16FP32Det.Init(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, tiling_data, &pipe);
        opBF16FP32Det.Process();
    } else if (TILING_KEY_IS(GROUP_NORM_GRAD_EMPTY_TENSOR_KEY)) {
        GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
        GET_TILING_DATA_WITH_STRUCT(GroupNormGradEmptyTilingData, tiling_data_in, tilingdata);
        const GroupNormGradEmptyTilingData* __restrict tiling_data = &tiling_data_in;
        TPipe pipe;
        EmptyDgamma<2> opDG(&pipe, tiling_data);
        opDG.Init(dgamma, dbeta);
        opDG.Process();
    }
}
