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
 * \file fused_adam.cpp
 * \brief
 */
#include "kernel_operator_list_tensor_intf.h"
#include "fused_adam_regbase.h"
#include "fused_adam_tiling_data.h"

using namespace AscendC;
using namespace FusedAdam;

extern "C" __global__ __aicore__ void fused_adam(GM_ADDR params, GM_ADDR grads, GM_ADDR exp_avgs, GM_ADDR exp_avg_sqs,
                                                 GM_ADDR max_exp_avg_sqs, GM_ADDR state_steps, GM_ADDR grad_scale,
                                                 GM_ADDR found_inf, GM_ADDR params_ref, GM_ADDR grads_ref,
                                                 GM_ADDR exp_avgs_ref, GM_ADDR exp_avg_sqs_ref,
                                                 GM_ADDR max_exp_avg_sqs_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(FusedAdamTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedAdamTilingData, tilingData, tiling);

    bool amsgrad_ = tilingData.amsgrad == 1 ? true : false;

    if (TILING_KEY_IS(0)) {
        if (amsgrad_) {
            FusedAdamKernelRegBase<float, true> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        } else {
            FusedAdamKernelRegBase<float, false> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        if (amsgrad_) {
            FusedAdamKernelRegBase<half, true> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        } else {
            FusedAdamKernelRegBase<half, false> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if (amsgrad_) {
            FusedAdamKernelRegBase<bfloat16_t, true> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        } else {
            FusedAdamKernelRegBase<bfloat16_t, false> op(&pipe);
            op.Init(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                    params_ref, grads_ref, exp_avgs_ref, exp_avg_sqs_ref, max_exp_avg_sqs_ref, tilingData);
            op.Process();
        }
    }
}
