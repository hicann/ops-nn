/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H

#include "kernel_operator.h"
#include "sigmoid_cross_entropy_with_logits_grad_v2_tiling_data.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FP32_BUF_NUM = 5;

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
class KernelSigmoidCrossEntropyWithLogitsGradV2 {
public:
    __aicore__ inline KernelSigmoidCrossEntropyWithLogitsGradV2()
    {}

    __aicore__ inline void Init(
        GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR gradient, GM_ADDR weight, GM_ADDR pos_weight,
        const SigmoidCrossEntropyWithLogitsGradV2TilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void SetProcessDataNum(int32_t index, int32_t loopCount);

private:
    TPipe pipe;
    SigmoidCrossEntropyWithLogitsGradV2TilingData tiling;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePredict;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTarget;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDout;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWeight;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePosWeight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGrad;

    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<T> predictGm, targetGm, doutGm, weightGm, posWeightGm, gradGm;

    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
};

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_V2_H
