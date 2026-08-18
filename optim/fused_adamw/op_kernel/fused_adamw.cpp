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
 * \file fused_adamw.cpp
 * \brief
 */
#include "kernel_operator_list_tensor_intf.h"
#include "fused_adamw_kernel.h"

using namespace AscendC;
using namespace FusedAdamW;

#ifdef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void fused_adamw(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                                  GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale,
                                                  GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR exp_avgs_ref,
                                                  GM_ADDR exp_avg_sqs_ref, GM_ADDR max_exp_avg_sqs_ref,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(FusedAdamWTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedAdamWTilingData, tilingData, tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint64_t tensorStart = static_cast<uint64_t>(blockIdx) * tilingData.tensorsPerCore;
    uint64_t perCoreTensorNum = tilingData.tensorsPerCore;
    if (blockIdx == (tilingData.usedRealCoreNum - 1)) {
        perCoreTensorNum = tilingData.lastCoreTensor;
    }
    uint64_t tensorEnd = tensorStart + perCoreTensorNum;
    if (tensorEnd > tilingData.tensorNum) {
        tensorEnd = tilingData.tensorNum;
    }
    if (TILING_KEY_IS(1)) {
        FusedAdamWKernel<half> op(&pipe);
        op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
                exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FusedAdamWKernel<bfloat16_t> op(&pipe);
        op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
                exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
        op.Process();
    } else {
        FusedAdamWKernel<float> op(&pipe);
        op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
                exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
        op.Process();
    }
}
#else
extern "C" __global__ __aicore__ void fused_adamw(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                                  GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale,
                                                  GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR exp_avgs_ref,
                                                  GM_ADDR exp_avg_sqs_ref, GM_ADDR max_exp_avg_sqs_ref,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(FusedAdamWTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedAdamWTilingData, tilingData, tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint64_t tensorStart = static_cast<uint64_t>(blockIdx) * tilingData.tensorsPerCore;
    uint64_t perCoreTensorNum = tilingData.tensorsPerCore;
    if (blockIdx == (tilingData.usedRealCoreNum - 1)) {
        perCoreTensorNum = tilingData.lastCoreTensor;
    }
    uint64_t tensorEnd = tensorStart + perCoreTensorNum;
    if (tensorEnd > tilingData.tensorNum) {
        tensorEnd = tilingData.tensorNum;
    }

#if (ORIG_DTYPE_PARAMS == DT_BF16)
    FusedAdamWKernel<bfloat16_t> op(&pipe);
    op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
            exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
    op.Process();
#elif (ORIG_DTYPE_PARAMS == DT_FLOAT16)
    FusedAdamWKernel<half> op(&pipe);
    op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
            exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
    op.Process();
#elif (ORIG_DTYPE_PARAMS == DT_FLOAT32)
    FusedAdamWKernel<float> op(&pipe);
    op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, params_ref,
            exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData, tensorStart, tensorEnd);
    op.Process();
#endif
}
#endif
