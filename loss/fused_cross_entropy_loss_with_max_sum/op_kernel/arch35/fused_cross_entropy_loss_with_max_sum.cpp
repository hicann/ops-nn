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
 * \file fused_cross_entropy_loss_with_max_sum.cpp
 * \brief FusedCrossEntropyLossWithMaxSum arch35(regbase) kernel entry
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "fused_cross_entropy_loss_with_max_sum_tiling_data.h"
#include "fused_cross_entropy_loss_with_max_sum_ar.h"

using namespace AscendC;
using namespace FusedCrossEntropyLossWithMaxSumOps;

namespace {
#define TILINGKEY_FULL 0   // 完整路径：loss + softmax
#define TILINGKEY_MEMORY 1 // 省显存路径：仅loss
} // namespace

#define FUSED_CE_MAX_SUM_FULL_IMPL(VOCAB_TYPE)                                                             \
    do {                                                                                                   \
        FusedCrossEntropyLossWithMaxSumRegBase<VOCAB_TYPE, true> op;                                       \
        op.Init(logits_max, sum_exp_logits, predicted_logits, vocab_parallel_logits, loss, softmax_logits, \
                &tilingData, &pipe);                                                                       \
        op.Process();                                                                                      \
    } while (0)

extern "C" __global__ __aicore__ void fused_cross_entropy_loss_with_max_sum(
    GM_ADDR logits_max, GM_ADDR sum_exp_logits, GM_ADDR predicted_logits, GM_ADDR input, GM_ADDR weight,
    GM_ADDR vocab_parallel_logits, GM_ADDR loss, GM_ADDR softmax_logits, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(FusedCrossEntropyLossWithMaxSumRegBaseTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TILINGKEY_FULL)) {
        // 单二进制覆盖三种vocab dtype，按tiling字段运行时分发（simplifiedKey不含optional输入dtype，
        // 多二进制无法被运行时区分，与910b的TILING_KEY_IS设计同理）
        if (tilingData.vocabDtypeId == FUSED_CE_MAX_SUM_DTYPE_FP32) {
            FUSED_CE_MAX_SUM_FULL_IMPL(float);
        } else if (tilingData.vocabDtypeId == FUSED_CE_MAX_SUM_DTYPE_FP16) {
            FUSED_CE_MAX_SUM_FULL_IMPL(half);
        } else {
            FUSED_CE_MAX_SUM_FULL_IMPL(bfloat16_t);
        }
    } else if (TILING_KEY_IS(TILINGKEY_MEMORY)) {
        // 省显存路径：仅计算 loss = log(sum_exp) - predicted，不涉及 vocab_parallel_logits。
        // sum_exp / predicted / loss 均为 fp32，因此 VocabType 固定为 float。
        FusedCrossEntropyLossWithMaxSumRegBase<float, false> op;
        op.Init(logits_max, sum_exp_logits, predicted_logits, vocab_parallel_logits, loss, softmax_logits, &tilingData,
                &pipe);
        op.Process();
    }
}
