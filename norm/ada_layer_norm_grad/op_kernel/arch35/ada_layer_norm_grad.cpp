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
 * \brief A5 (ascend950) specific kernel entry for AdaLayerNormGrad.
 */
#include "ada_layer_norm_grad_cut_n_arch35.h"
#include "ada_layer_norm_grad_cut_d_arch35.h"

using namespace AdaLayerNormGrad;

extern "C" __global__ __aicore__ void ada_layer_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean,
                                                          GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta, GM_ADDR pd_x,
                                                          GM_ADDR pd_scale, GM_ADDR pd_shift, GM_ADDR pd_gamma,
                                                          GM_ADDR pd_beta, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(AdaLayerNormGradTilingDataArch35, tiling_data, tiling);

#define INIT_CUT(op)                                                                                         \
    op.Init(dy, x, rstd, mean, scale, gamma, beta, pd_x, pd_scale, pd_shift, pd_gamma, pd_beta, tiling_data, \
            usrWorkspace);                                                                                   \
    op.Process()

    if (TILING_KEY_IS(110)) {
        KernelCutN<float, float, 110> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(120)) {
        KernelCutN<half, half, 120> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(130)) {
        KernelCutN<half, float, 130> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(140)) {
        KernelCutN<bfloat16_t, bfloat16_t, 140> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(150)) {
        KernelCutN<bfloat16_t, float, 150> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(210)) {
        KernelCutN<float, float, 210> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(220)) {
        KernelCutN<half, half, 220> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(230)) {
        KernelCutN<half, float, 230> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(240)) {
        KernelCutN<bfloat16_t, bfloat16_t, 240> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(250)) {
        KernelCutN<bfloat16_t, float, 250> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(310)) {
        KernelCutD<float, float, 310> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(320)) {
        KernelCutD<half, half, 320> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(330)) {
        KernelCutD<half, float, 330> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(340)) {
        KernelCutD<bfloat16_t, bfloat16_t, 340> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(350)) {
        KernelCutD<bfloat16_t, float, 350> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(410)) {
        KernelCutD<float, float, 410> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(420)) {
        KernelCutD<half, half, 420> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(430)) {
        KernelCutD<half, float, 430> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(440)) {
        KernelCutD<bfloat16_t, bfloat16_t, 440> op;
        INIT_CUT(op);
    } else if (TILING_KEY_IS(450)) {
        KernelCutD<bfloat16_t, float, 450> op;
        INIT_CUT(op);
    }
}
