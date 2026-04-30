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
 * \file sigmoid_cross_entropy_with_logits_v2.h
 * \brief
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H

#include "kernel_operator.h"
#include "sigmoid_cross_entropy_with_logits_v2_tiling_data.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FP32_BUF_NUM = 6;

template <typename T>
class KernelSigmoidCrossEntropyWithLogitsV2 {
public:
    __aicore__ inline KernelSigmoidCrossEntropyWithLogitsV2()
    {}

    __aicore__ inline void Init(
        GM_ADDR predict, GM_ADDR target, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR loss,
        SigmoidCrossEntropyWithLogitsV2TilingData tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t GetCurTileLength(int32_t index) const;
    __aicore__ inline void CopyIn(int32_t index, uint32_t curLen);
    __aicore__ inline void Compute(int32_t index, uint32_t curLen);
    __aicore__ inline void CopyOut(int32_t index, uint32_t curLen);

private:
    TPipe pipe;
    SigmoidCrossEntropyWithLogitsV2TilingData tiling;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePredict, inQueueTarget;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWeight, inQueuePosWeight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueLoss;

    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<T> predictGm, targetGm, weightGm, posWeightGm;
    GlobalTensor<float> lossGm;

    uint32_t coreOffset = 0;
    uint32_t coreLength = 0;
};

#endif // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_V2_H
