/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SMOOTH_L1_LOSS_H
#define SMOOTH_L1_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "smooth_l1_loss_tilingdata.h"

namespace SmoothL1Loss {

using namespace AscendC;

template <typename T>
class KernelSmoothL1Loss {
    static constexpr int32_t BUFFER_NUM = 1;
    static constexpr bool NEED_CAST = !std::is_same<T, float>::value;

public:
    __aicore__ inline KernelSmoothL1Loss() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR loss, const SmoothL1LossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueuePredict_;
    TQue<QuePosition::VECIN, 1> inQueueLabel_;
    TQue<QuePosition::VECOUT, 1> outQueueLoss_;

    TBuf<QuePosition::VECCALC> workBufTmp_;
    TBuf<QuePosition::VECCALC> workBufSigma_;
    TBuf<QuePosition::VECCALC> workBufMask_;
    TBuf<QuePosition::VECCALC> workBufCast_;

    GlobalTensor<T> predictGm_;
    GlobalTensor<T> labelGm_;
    GlobalTensor<T> lossGm_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float sigma_ = 1.0f;
    float multiplyValue_ = 1.0f;
    float addsValue_ = -0.5f;
};

template <typename T>
__aicore__ inline void KernelSmoothL1Loss<T>::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR loss,
                                                   const SmoothL1LossTilingData* tilingData)
{
    int64_t startOffset = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalNum - startOffset;
    if (remaining <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    blockLength_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;
    ubLength_ = tilingData->ubFactor;
    sigma_ = tilingData->Sigma;
    multiplyValue_ = tilingData->MultiplyValue;
    addsValue_ = tilingData->AddsValue;

    predictGm_.SetGlobalBuffer((__gm__ T*)predict + startOffset, blockLength_);
    labelGm_.SetGlobalBuffer((__gm__ T*)label + startOffset, blockLength_);
    lossGm_.SetGlobalBuffer((__gm__ T*)loss + startOffset, blockLength_);

    int64_t typeSize = sizeof(T);
    int64_t computeTypeSize = sizeof(float);
    int64_t alignElements = 256 / typeSize;

    if (ubLength_ < alignElements) {
        ubLength_ = alignElements;
    }

    int64_t allocElems = ubLength_;
    int64_t cmpBufSize = ((allocElems / 8 + 255) / 256) * 256;
    if (cmpBufSize < 256) {
        cmpBufSize = 256;
    }

    pipe_.InitBuffer(inQueuePredict_, 1, allocElems * typeSize);
    pipe_.InitBuffer(inQueueLabel_, 1, allocElems * typeSize);
    pipe_.InitBuffer(outQueueLoss_, 1, allocElems * typeSize);
    pipe_.InitBuffer(workBufTmp_, allocElems * computeTypeSize);
    pipe_.InitBuffer(workBufSigma_, allocElems * computeTypeSize);
    pipe_.InitBuffer(workBufMask_, cmpBufSize);
    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(workBufCast_, allocElems * computeTypeSize);
    }
}

template <typename T>
__aicore__ inline void KernelSmoothL1Loss<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    int64_t loopTimes = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopTimes; i++) {
        int64_t currentNum = (blockLength_ - i * ubLength_) > ubLength_ ? ubLength_ : (blockLength_ - i * ubLength_);
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void KernelSmoothL1Loss<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    auto predictLocal = inQueuePredict_.template AllocTensor<T>();
    auto labelLocal = inQueueLabel_.template AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(predictLocal, predictGm_[progress * ubLength_], copyParams, padParams);
    DataCopyPad(labelLocal, labelGm_[progress * ubLength_], copyParams, padParams);
    inQueuePredict_.EnQue(predictLocal);
    inQueueLabel_.EnQue(labelLocal);
}

template <typename T>
__aicore__ inline void KernelSmoothL1Loss<T>::Compute(int64_t currentNum)
{
    auto predictLocal = inQueuePredict_.template DeQue<T>();
    auto labelLocal = inQueueLabel_.template DeQue<T>();
    auto lossLocal = outQueueLoss_.template AllocTensor<T>();

    uint32_t elemCount = static_cast<uint32_t>(currentNum);
    constexpr uint32_t COMPARE_ALIGN_ELEMENTS = 256 / static_cast<uint32_t>(sizeof(float));
    uint32_t alignedCount = (elemCount + COMPARE_ALIGN_ELEMENTS - 1) / COMPARE_ALIGN_ELEMENTS * COMPARE_ALIGN_ELEMENTS;

    LocalTensor<float> tmpTensor = workBufTmp_.Get<float>();
    LocalTensor<float> sigmaTensor = workBufSigma_.Get<float>();
    LocalTensor<uint8_t> maskTensor = workBufMask_.Get<uint8_t>();

    if constexpr (NEED_CAST) {
        LocalTensor<float> predictFp32 = workBufCast_.Get<float>();
        Cast(predictFp32, predictLocal, RoundMode::CAST_NONE, elemCount);
        LocalTensor<float> labelFp32 = sigmaTensor;
        Cast(labelFp32, labelLocal, RoundMode::CAST_NONE, elemCount);

        LocalTensor<float> absTensor = tmpTensor;
        Sub(absTensor, predictFp32, labelFp32, elemCount);
        Abs(absTensor, absTensor, elemCount);

        LocalTensor<float> quadTensor = predictFp32;
        Muls(quadTensor, absTensor, static_cast<float>(multiplyValue_), elemCount);
        Mul(quadTensor, quadTensor, absTensor, elemCount);

        Duplicate(sigmaTensor, static_cast<float>(sigma_), elemCount);
        Compare(maskTensor, absTensor, sigmaTensor, CMPMODE::LT, alignedCount);

        Adds(absTensor, absTensor, static_cast<float>(addsValue_), elemCount);

        LocalTensor<float> resultFp32 = sigmaTensor;
        Select(resultFp32, maskTensor, quadTensor, absTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(lossLocal, resultFp32, RoundMode::CAST_RINT, elemCount);
        } else {
            Cast(lossLocal, resultFp32, RoundMode::CAST_NONE, elemCount);
        }
    } else {
        LocalTensor<float> predictFp32 = predictLocal.template ReinterpretCast<float>();
        LocalTensor<float> labelFp32 = labelLocal.template ReinterpretCast<float>();

        LocalTensor<float> absTensor = tmpTensor;
        Sub(absTensor, predictFp32, labelFp32, elemCount);
        Abs(absTensor, absTensor, elemCount);

        LocalTensor<float> quadTensor = lossLocal.template ReinterpretCast<float>();
        Muls(quadTensor, absTensor, static_cast<float>(multiplyValue_), elemCount);
        Mul(quadTensor, quadTensor, absTensor, elemCount);

        Duplicate(sigmaTensor, static_cast<float>(sigma_), elemCount);
        Compare(maskTensor, absTensor, sigmaTensor, CMPMODE::LT, alignedCount);

        Adds(absTensor, absTensor, static_cast<float>(addsValue_), elemCount);

        LocalTensor<float> resultFp32 = sigmaTensor;
        Select(resultFp32, maskTensor, quadTensor, absTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);
        LocalTensor<float> lossFp32 = lossLocal.template ReinterpretCast<float>();
        Adds(lossFp32, resultFp32, static_cast<float>(0), elemCount);
    }

    inQueuePredict_.FreeTensor(predictLocal);
    inQueueLabel_.FreeTensor(labelLocal);
    outQueueLoss_.EnQue(lossLocal);
}

template <typename T>
__aicore__ inline void KernelSmoothL1Loss<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    auto lossLocal = outQueueLoss_.template DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(lossGm_[progress * ubLength_], lossLocal, copyParams);
    outQueueLoss_.FreeTensor(lossLocal);
}

} // namespace SmoothL1Loss

#endif // SMOOTH_L1_LOSS_H
