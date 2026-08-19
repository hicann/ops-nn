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
 * \file add_layer_norm_grad.cpp
 * \brief A5 (ascend910_95) specific kernel entry for AddLayerNormGrad
 */

#include "add_layer_norm_grad_cut_n.h"
#include "add_layer_norm_grad_cut_d.h"

using namespace AscendC;
using namespace AddLayerNormGrad;

extern "C" __global__ __aicore__ void add_layer_norm_grad(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd,
                                                          GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum, GM_ADDR d_x,
                                                          GM_ADDR d_gamma, GM_ADDR d_beta, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);

#define INIT_CUT_N_PROCESS                                                                           \
    op.Init(dy, x_1, x_2, rstd, mean, gamma, dsum, d_x, d_gamma, d_beta, tiling_data, usrWorkspace); \
    op.CutNProcess()
#define INIT_CUT_D_PROCESS                                                                           \
    op.Init(dy, x_1, x_2, rstd, mean, gamma, dsum, d_x, d_gamma, d_beta, tiling_data, usrWorkspace); \
    op.CutDProcess()

    if (TILING_KEY_IS(10)) {
        KernelAddLayerNormGradA35<float, 10> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(11)) {
        KernelAddLayerNormGradA35<float, 11> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(20)) {
        KernelAddLayerNormGradA35<half, 20> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        KernelAddLayerNormGradA35<half, 21> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(30)) {
        KernelAddLayerNormGradA35<bfloat16_t, 30> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(31)) {
        KernelAddLayerNormGradA35<bfloat16_t, 31> op;
        INIT_CUT_N_PROCESS;
    } else if (TILING_KEY_IS(40)) {
        KernelAddLayerNormGradLargeA35<float, 40> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(41)) {
        KernelAddLayerNormGradLargeA35<float, 41> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(50)) {
        KernelAddLayerNormGradLargeA35<half, 50> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(51)) {
        KernelAddLayerNormGradLargeA35<half, 51> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(60)) {
        KernelAddLayerNormGradLargeA35<bfloat16_t, 60> op;
        INIT_CUT_D_PROCESS;
    } else if (TILING_KEY_IS(61)) {
        KernelAddLayerNormGradLargeA35<bfloat16_t, 61> op;
        INIT_CUT_D_PROCESS;
    }
}
