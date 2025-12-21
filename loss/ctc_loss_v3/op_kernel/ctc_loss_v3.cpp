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
 * \file ctc_loss_v3.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v3.h"
using namespace CTCLossV3NS;

extern "C" __global__ __aicore__ void ctc_loss_v3(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths, GM_ADDR neg_log_likelihood,
    GM_ADDR log_alpha, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    CTCLossV3<DTYPE_LOG_PROBS, DTYPE_TARGETS> op;
    TPipe pipe;
    op.Init(
        &tilingData, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
        &pipe);
    op.Process();
}