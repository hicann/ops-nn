/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctc_loss_v2.cpp
 * \brief ctc_loss_v2
 */
#include "arch35/ctc_loss_v2_fp32.h"
#include "arch35/ctc_loss_v2_fp16.h"
#include "arch35/ctc_loss_v2_tiling_key.h"

using namespace AscendC;
using namespace CTCLossV2;

template <uint64_t isFP32, uint64_t threadTypeInt32>
__global__ __aicore__ void ctc_loss_v2(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths, GM_ADDR neg_log_likelihood,
    GM_ADDR log_alpha, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (isFP32 == CTC_LOSS_V2_TPL_KEY_TRUE) {
        if constexpr (threadTypeInt32 == CTC_LOSS_V2_TPL_KEY_TRUE) {
            CTCLossV2::CTCLossV2FP32<DTYPE_LOG_PROBS, DTYPE_TARGET_LENGTHS, int32_t> op;
            op.Init(
                log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
                &tilingData, &pipe);
            op.Process();
        } else {
            CTCLossV2::CTCLossV2FP32<DTYPE_LOG_PROBS, DTYPE_TARGET_LENGTHS, int64_t> op;
            op.Init(
                log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
                &tilingData, &pipe);
            op.Process();
        }
    } else {
        if constexpr (threadTypeInt32 == CTC_LOSS_V2_TPL_KEY_TRUE) {
            CTCLossV2::CTCLossV2FP16<DTYPE_LOG_PROBS, DTYPE_TARGET_LENGTHS, int32_t> op;
            op.Init(
                log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
                &tilingData);
            op.Process();
        } else {
            CTCLossV2::CTCLossV2FP16<DTYPE_LOG_PROBS, DTYPE_TARGET_LENGTHS, int64_t> op;
            op.Init(
                log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
                &tilingData);
            op.Process();
        }
    }
}