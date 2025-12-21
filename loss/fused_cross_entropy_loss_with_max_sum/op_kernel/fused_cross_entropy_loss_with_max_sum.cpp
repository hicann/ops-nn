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
 * \file fused_cross_entropy_loss_with_max_sum.cpp
 * \brief
 */

#include "fused_cross_entropy_loss_with_max_sum.h"

using namespace AscendC;
// kernel function
extern "C" __global__ __aicore__ void fused_cross_entropy_loss_with_max_sum(
    GM_ADDR logits_max, GM_ADDR sum_exp_logits, GM_ADDR predicted_logits, GM_ADDR input, GM_ADDR weight,
    GM_ADDR vocab_parallel_logits, GM_ADDR loss, GM_ADDR softmax_logits, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[6] = {vocab_parallel_logits, logits_max, sum_exp_logits, predicted_logits, softmax_logits, loss};
    if (TILING_KEY_IS(0)) {
        FusedCrossEntropyLossWithMaxSum<float> op(gmTensor, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        FusedCrossEntropyLossWithMaxSum<half> op(gmTensor, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FusedCrossEntropyLossWithMaxSum<bfloat16_t> op(gmTensor, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        FusedCrossEntropyLossWithMaxSum<float> op(gmTensor, tilingData, pipe);
        op.ProcessForMemory();
    }
}
