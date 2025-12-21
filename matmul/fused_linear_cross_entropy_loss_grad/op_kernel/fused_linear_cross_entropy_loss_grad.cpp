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
 * \file fused_linear_cross_entropy_loss_grad.cpp
 * \brief
 */
#include "fused_linear_cross_entropy_loss_grad_config.h"
#include "fused_linear_cross_entropy_loss_grad_high_perf.h"
#include "fused_linear_cross_entropy_loss_grad_mem_friendly.h"
#ifdef __CCE_KT_TEST__
#include "fused_linear_cross_entropy_loss_grad_tiling_data_def.h"
#endif  // __CCE_KT_TEST__

using namespace FusedLinearCrossEntropyLossGradNS;
using namespace matmul;

extern "C" __global__ __aicore__ void fused_linear_cross_entropy_loss_grad(
    GM_ADDR grad, GM_ADDR input, GM_ADDR weight, GM_ADDR target_mask, GM_ADDR masked_target,
    GM_ADDR logits_max, GM_ADDR sum_exp_logits, GM_ADDR softmax,
    GM_ADDR input_grad_out, GM_ADDR weight_grad_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if TILING_KEY_IS(HIGH_PERF_KEY) {
        COPY_TILING_WITH_STRUCT(FusedLinearCrossEntropyLossGradHighPerfTilingData, tiling, tdp);
        using MM = MatmulImpl<
            MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, DTYPE_INPUT, true>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MM_MDL_CFG>;
        
        FusedLinearCrossEntropyLossGradHighPerf<MM, DTYPE_INPUT, DTYPE_TARGET_MASK, DTYPE_MASKED_TARGET> op(*tdp);
        op.Init(input, weight, softmax, target_mask, masked_target, grad, input_grad_out, weight_grad_out, workspace);
        op.Process();
    } else if (TILING_KEY_IS(MEM_FRIENDLY_KEY)) {
        GET_TILING_DATA_PTR_WITH_STRUCT(FusedLinearCrossEntropyLossGradMemFriendlyTilingData, tdp, tiling);
        using MM0 = MatmulImpl<
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT, true>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MM_MDL_CFG>;
        using MM1 = MatmulImpl<
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MM_MDL_CFG>;
        using MM2 = MatmulImpl<
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT, true>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_INPUT>,
            MM_MDL_CFG>;
    
        FusedLinearCrossEntropyLossGradMemFriendly<DTYPE_INPUT, DTYPE_TARGET_MASK, DTYPE_MASKED_TARGET, MM0, MM1, MM2> op;
        op.Init(
            grad, input, weight, target_mask, masked_target,
            logits_max, sum_exp_logits, input_grad_out, weight_grad_out,
            workspace, tdp
        );
        op.Process();
    }

    #ifdef __CCE_KT_TEST__
    EmptyTestFunc();
    #endif  // __CCE_KT_TEST__
}