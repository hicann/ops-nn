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
 * \file swi_glu_grad_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/swi_glu_grad_regbase_tiling.h"
#include "arch35/swi_glu_grad_move_align.h"
#include "arch35/swi_glu_grad_ub_rearrange.h"

using namespace AscendC;
using namespace SwiGluGrad;

#define SWIGLUGRAD_MOVEALIGN_FP16        1000001
#define SWIGLUGRAD_MOVEALIGN_FP32        1000011
#define SWIGLUGRAD_MOVEALIGN_BF16        1000021
#define SWIGLUGRAD_UB_REARRANGE_FP16     1000000
#define SWIGLUGRAD_UB_REARRANGE_FP32     1000010
#define SWIGLUGRAD_UB_REARRANGE_BF16     1000020

extern "C" __global__ __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                    
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(GluBaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(SWIGLUGRAD_MOVEALIGN_FP32)) {
        SwiGluGradMoveAlignKernel<float> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    } else if (TILING_KEY_IS(SWIGLUGRAD_MOVEALIGN_FP16)) {
        SwiGluGradMoveAlignKernel<half> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    } else if (TILING_KEY_IS(SWIGLUGRAD_MOVEALIGN_BF16)) {
        SwiGluGradMoveAlignKernel<bfloat16_t> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    } else if (TILING_KEY_IS(SWIGLUGRAD_UB_REARRANGE_FP32)) {
        SwiGluGradUbRearrangeKernel<float, uint32_t> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    } else if (TILING_KEY_IS(SWIGLUGRAD_UB_REARRANGE_FP16)) {
        SwiGluGradUbRearrangeKernel<half, uint16_t> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    } else if (TILING_KEY_IS(SWIGLUGRAD_UB_REARRANGE_BF16)) {
        SwiGluGradUbRearrangeKernel<bfloat16_t, uint16_t> obj(pipe);
        obj.Init(gradout_gm, input_gm, output_gm, tilingData);
        obj.Process();
    }
}