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
 * \file ctc_loss_v2_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/ctc_loss_v2_grad.h"
using namespace CTCLossV2GradNS;

#define TILING_KEY_INT32 0
#define TILING_KEY_INT64 1

extern "C" __global__ __aicore__ void ctc_loss_v2_grad(
    GM_ADDR grad_out, GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
    GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR grad, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(TILING_KEY_INT32)) {
        GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
        CTCLossV2GradNS::CTCLossV2Grad<DTYPE_GRAD, DTYPE_TARGETS, int32_t> op;
        op.Init(
            grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad, workspace,
            &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_INT64)) {
        GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
        CTCLossV2GradNS::CTCLossV2Grad<DTYPE_GRAD, DTYPE_TARGETS, int64_t> op;
        op.Init(
            grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, grad, workspace,
            &tilingData);
        op.Process();
    }
}