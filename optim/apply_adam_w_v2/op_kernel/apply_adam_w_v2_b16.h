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
 * \file apply_adam_w_v2_b16.h
 * \brief
 */

#ifndef APPLYADAM_W_V2_B16_H
#define APPLYADAM_W_V2_B16_H

#include "apply_adam_w_v2_base.h"

namespace ApplyAdamWV2 {
using namespace AscendC;

template <typename T, typename U>
class ApplyAdamWV2B16 {
public:
    __aicore__ inline ApplyAdamWV2B16(){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad, GM_ADDR step,
                                GM_ADDR maxGradNorm, GM_ADDR workspace, const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ParseTilingData(const ApplyAdamWV2TilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount);
    __aicore__ inline void Compute(int32_t dataCount);
    __aicore__ inline void CopyOut(int64_t index, int64_t dataCount);
    __aicore__ inline float ScalarPowB16(float x, float y);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB16_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueB16_;
    TBuf<QuePosition::VECCALC> inCastBufB16_;
    TBuf<QuePosition::VECCALC> outCastBufB16_;

    TBuf<QuePosition::VECCALC> powTempBuf1B16_;
    TBuf<QuePosition::VECCALC> powTempBuf2B16_;

    GlobalTensor<T> gmVarB16_;
    GlobalTensor<T> gmExpAvgB16_;
    GlobalTensor<T> gmExpAvgSqB16_;
    GlobalTensor<T> gmMaxGradNormB16_;
    GlobalTensor<T> gmGradB16_;
    GlobalTensor<U> gmStepB16_;

    float stepB16_ = 0;

    int64_t numPerLoopB16_ = 0;
    int64_t loopNumPerCoreB16_ = 0;
    int64_t numLastLoopB16_ = 0;
    int64_t handleExtraLoopCoreNumB16_ = 0;
    int64_t usedCoreNum_ = 0;
    bool amsgradB16_ = false;
    float beta1_ = 0;
    float beta2_ = 0;
    float lrB16_ = 0;
    float weightDecayB16_ = 0;
    float eps_ = 0;
    bool maximizeB16_ = false;
    bool isBfloat16_ = false;

    float realWeightDecayB16_ = 0;
    float stepSizeB16_ = 0;
    float biasCorrection2SqrtB16_ = 0;
    float oneSubBeta1_ = 0;
    float oneSubBeta2_ = 0;
    float realBeta2B16_ = 0;
    float negOne_ = -1.0f;
    float realEps_ = 0;

    int64_t varOffsetB16_ = 0;
    int64_t expAvgOffsetB16_ = 0;
    int64_t expAvgSqOffsetB16_ = 0;
    int64_t gradOffset_ = 0;
    int64_t maxGradNormOffsetB16_ = 0;
    int64_t maxGradOutOffsetB16_ = 0;
    int64_t blockIdx_ = GetBlockIdx();
};

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::Init(GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad,
                                                   GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace,
                                                   const ApplyAdamWV2TilingData* tilingData)
{
    this->ParseTilingData(tilingData);
    gmStepB16_.SetGlobalBuffer((__gm__ U*)step, 1);
    stepB16_ = static_cast<float>(gmStepB16_.GetValue(0));
    int64_t gmOffset = blockIdx_ * numPerLoopB16_;

    gmVarB16_.SetGlobalBuffer((__gm__ T*)var + gmOffset);
    gmExpAvgB16_.SetGlobalBuffer((__gm__ T*)expAvg + gmOffset);
    gmExpAvgSqB16_.SetGlobalBuffer((__gm__ T*)expAvgSq + gmOffset);
    gmGradB16_.SetGlobalBuffer((__gm__ T*)grad + gmOffset);

    pipe_.InitBuffer(inQueueB16_, BUFFER_NUM, IN_BUFFER_NUM * numPerLoopB16_ * sizeof(T));
    pipe_.InitBuffer(outQueueB16_, BUFFER_NUM, OUT_BUFFER_NUM * numPerLoopB16_ * sizeof(T));
    pipe_.InitBuffer(inCastBufB16_, IN_BUFFER_NUM * numPerLoopB16_ * sizeof(float));
    pipe_.InitBuffer(outCastBufB16_, OUT_BUFFER_NUM * numPerLoopB16_ * sizeof(float));

    pipe_.InitBuffer(powTempBuf1B16_, BYTE_ONE_BLOCK);
    pipe_.InitBuffer(powTempBuf2B16_, BYTE_ONE_BLOCK);

    if (amsgradB16_) {
        gmMaxGradNormB16_.SetGlobalBuffer((__gm__ T*)maxGradNorm + gmOffset);
    }

    stepB16_ += 1;
    float biasCorrection1 = 1.0f - ScalarPowB16(beta1_, stepB16_);
    float biasCorrection2 = 1.0f - ScalarPowB16(beta2_, stepB16_);

    stepSizeB16_ = lrB16_ / biasCorrection1;
    biasCorrection2SqrtB16_ = 1.0f / sqrt(biasCorrection2);

    realWeightDecayB16_ = 1.0f - lrB16_ * weightDecayB16_;
    oneSubBeta1_ = 1.0f - beta1_;
    oneSubBeta2_ = 1.0f - beta2_;
    realBeta2B16_ = beta2_;
    realEps_ = eps_;
    varOffsetB16_ = VAR_ORDER_IN_LOCAL_TENSOR * numPerLoopB16_;
    expAvgOffsetB16_ = EXP_AVG_ORDER_IN_LOCAL_TENSOR * numPerLoopB16_;
    expAvgSqOffsetB16_ = EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR * numPerLoopB16_;
    gradOffset_ = GRAD_NORM_ORDER_IN_LOCAL_TENSOR * numPerLoopB16_;
    maxGradNormOffsetB16_ = MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR * numPerLoopB16_;
    maxGradOutOffsetB16_ = MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR * numPerLoopB16_;
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::ParseTilingData(const ApplyAdamWV2TilingData* tilingData)
{
    numPerLoopB16_ = tilingData->numPerLoop;
    loopNumPerCoreB16_ = tilingData->loopNumPerCore;
    numLastLoopB16_ = tilingData->numLastLoop;
    usedCoreNum_ = tilingData->usedCoreNum;
    handleExtraLoopCoreNumB16_ = tilingData->handleExtraLoopCoreNum;
    beta1_ = tilingData->beta1;
    beta2_ = tilingData->beta2;
    lrB16_ = tilingData->lr;
    weightDecayB16_ = tilingData->weightDecay;
    eps_ = tilingData->eps;

    if (tilingData->amsgrad != 0) {
        amsgradB16_ = true;
    }

    if (tilingData->maximize != 0) {
        maximizeB16_ = true;
    }

    if (tilingData->isBfloat16 != 0) {
        isBfloat16_ = true;
    }
}

template <typename T, typename U>
__aicore__ inline float ApplyAdamWV2B16<T, U>::ScalarPowB16(float x, float y)
{
    LocalTensor<float> baseLocal = powTempBuf1B16_.Get<float>();
    LocalTensor<float> outLocal = powTempBuf2B16_.Get<float>();
    PipeBarrier<PIPE_V>();
    Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
    PipeBarrier<PIPE_V>();
    Power<float, false>(outLocal, baseLocal, y, BLOCK_SIZE_FOR_FLOAT32);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float result = outLocal.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    ;
    return result;
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::CopyIn(int64_t index, int64_t dataCount)
{
    int64_t offset = usedCoreNum_ * index * numPerLoopB16_;
    LocalTensor<T> dataLocal = inQueueB16_.AllocTensor<T>();

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(dataLocal[varOffsetB16_], gmVarB16_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[expAvgOffsetB16_], gmExpAvgB16_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[expAvgSqOffsetB16_], gmExpAvgSqB16_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(dataLocal[gradOffset_], gmGradB16_[offset], dataCopyParams, dataCopyPadParams);

    if (amsgradB16_) {
        DataCopyPad(dataLocal[maxGradNormOffsetB16_], gmMaxGradNormB16_[offset], dataCopyParams, dataCopyPadParams);
    }
    inQueueB16_.EnQue(dataLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::Compute(int32_t dataCount)
{
    LocalTensor<T> dataLocal = inQueueB16_.DeQue<T>();
    LocalTensor<T> dataOutLocal = outQueueB16_.AllocTensor<T>();
    LocalTensor<float> inCastLocal = inCastBufB16_.Get<float>();
    LocalTensor<float> outCastLocal = outCastBufB16_.Get<float>();

    Cast(inCastLocal[gradOffset_], dataLocal[gradOffset_], RoundMode::CAST_NONE, dataCount);
    Cast(inCastLocal[varOffsetB16_], dataLocal[varOffsetB16_], RoundMode::CAST_NONE, dataCount);
    Cast(inCastLocal[expAvgOffsetB16_], dataLocal[expAvgOffsetB16_], RoundMode::CAST_NONE, dataCount);
    Cast(inCastLocal[expAvgSqOffsetB16_], dataLocal[expAvgSqOffsetB16_], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    if (maximizeB16_) {
        // grad = -grad
        Muls(inCastLocal[gradOffset_], inCastLocal[gradOffset_], negOne_, dataCount);
    }
    // param.mul_(1 - lr * weight_decay)
    Muls(outCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], realWeightDecayB16_, dataCount);

    // exp_avg.lerp_(grad, 1 - beta1)
    PipeBarrier<PIPE_V>();
    Sub(outCastLocal[expAvgOffsetB16_], inCastLocal[gradOffset_], inCastLocal[expAvgOffsetB16_], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(outCastLocal[expAvgOffsetB16_], outCastLocal[expAvgOffsetB16_], oneSubBeta1_, dataCount);
    PipeBarrier<PIPE_V>();
    Add(outCastLocal[expAvgOffsetB16_], outCastLocal[expAvgOffsetB16_], inCastLocal[expAvgOffsetB16_], dataCount);

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    PipeBarrier<PIPE_V>();
    Muls(inCastLocal[expAvgSqOffsetB16_], inCastLocal[expAvgSqOffsetB16_], realBeta2B16_, dataCount);
    PipeBarrier<PIPE_V>();
    Mul(inCastLocal[gradOffset_], inCastLocal[gradOffset_], inCastLocal[gradOffset_], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inCastLocal[gradOffset_], inCastLocal[gradOffset_], oneSubBeta2_, dataCount);
    PipeBarrier<PIPE_V>();
    Add(outCastLocal[expAvgSqOffsetB16_], inCastLocal[expAvgSqOffsetB16_], inCastLocal[gradOffset_], dataCount);
    PipeBarrier<PIPE_V>();

    if (amsgradB16_) {
        PipeBarrier<PIPE_V>();
        Cast(inCastLocal[maxGradNormOffsetB16_], dataLocal[maxGradNormOffsetB16_], RoundMode::CAST_NONE, dataCount);
        // torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
        PipeBarrier<PIPE_V>();
        Max(outCastLocal[maxGradOutOffsetB16_], inCastLocal[maxGradNormOffsetB16_], outCastLocal[expAvgSqOffsetB16_],
            dataCount);
        PipeBarrier<PIPE_V>();
        Sqrt(inCastLocal[varOffsetB16_], outCastLocal[maxGradOutOffsetB16_], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(inCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], biasCorrection2SqrtB16_, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(inCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], realEps_, dataCount);
        PipeBarrier<PIPE_V>();
        if (isBfloat16_) {
            Cast(dataOutLocal[maxGradOutOffsetB16_], outCastLocal[maxGradOutOffsetB16_], RoundMode::CAST_ROUND,
                 dataCount);
        } else {
            Cast(dataOutLocal[maxGradOutOffsetB16_], outCastLocal[maxGradOutOffsetB16_], RoundMode::CAST_RINT,
                 dataCount);
        }
    } else {
        // denom = (exp_avg_sq.sqrt() / bias_corrections_sqrt) + eps
        PipeBarrier<PIPE_V>();
        Sqrt(inCastLocal[varOffsetB16_], outCastLocal[expAvgSqOffsetB16_], dataCount);
        PipeBarrier<PIPE_V>();
        Muls(inCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], biasCorrection2SqrtB16_, dataCount);
        PipeBarrier<PIPE_V>();
        Adds(inCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], realEps_, dataCount);
    }

    // param.addcdiv_(exp_avg, denom, value=-step_size)
    PipeBarrier<PIPE_V>();
    Div(inCastLocal[varOffsetB16_], outCastLocal[expAvgOffsetB16_], inCastLocal[varOffsetB16_], dataCount);
    PipeBarrier<PIPE_V>();
    Muls(inCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], stepSizeB16_, dataCount);
    PipeBarrier<PIPE_V>();
    Sub(outCastLocal[varOffsetB16_], outCastLocal[varOffsetB16_], inCastLocal[varOffsetB16_], dataCount);
    PipeBarrier<PIPE_V>();
    if (isBfloat16_) {
        Cast(dataOutLocal[varOffsetB16_], outCastLocal[varOffsetB16_], RoundMode::CAST_ROUND, dataCount);
        Cast(dataOutLocal[expAvgOffsetB16_], outCastLocal[expAvgOffsetB16_], RoundMode::CAST_ROUND, dataCount);
        Cast(dataOutLocal[expAvgSqOffsetB16_], outCastLocal[expAvgSqOffsetB16_], RoundMode::CAST_ROUND, dataCount);
    } else {
        Cast(dataOutLocal[varOffsetB16_], outCastLocal[varOffsetB16_], RoundMode::CAST_RINT, dataCount);
        Cast(dataOutLocal[expAvgOffsetB16_], outCastLocal[expAvgOffsetB16_], RoundMode::CAST_RINT, dataCount);
        Cast(dataOutLocal[expAvgSqOffsetB16_], outCastLocal[expAvgSqOffsetB16_], RoundMode::CAST_RINT, dataCount);
    }
    PipeBarrier<PIPE_V>();
    inQueueB16_.FreeTensor(dataLocal);
    outQueueB16_.EnQue(dataOutLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::CopyOut(int64_t index, int64_t dataCount)
{
    int64_t offset = usedCoreNum_ * index * numPerLoopB16_;
    LocalTensor<T> dataOutLocal = outQueueB16_.DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(gmVarB16_[offset], dataOutLocal[varOffsetB16_], copyParams);
    DataCopyPad(gmExpAvgB16_[offset], dataOutLocal[expAvgOffsetB16_], copyParams);
    DataCopyPad(gmExpAvgSqB16_[offset], dataOutLocal[expAvgSqOffsetB16_], copyParams);

    if (amsgradB16_) {
        DataCopyPad(gmMaxGradNormB16_[offset], dataOutLocal[maxGradOutOffsetB16_], copyParams);
    }
    outQueueB16_.FreeTensor(dataOutLocal);
}

template <typename T, typename U>
__aicore__ inline void ApplyAdamWV2B16<T, U>::Process()
{
    if (blockIdx_ < usedCoreNum_) {
        int64_t curLoopCount = loopNumPerCoreB16_;
        if (blockIdx_ < handleExtraLoopCoreNumB16_ - 1) {
            curLoopCount += 1;
        }

        for (int64_t n = 0; n < curLoopCount; n++) {
            CopyIn(n, numPerLoopB16_);
            Compute(numPerLoopB16_);
            CopyOut(n, numPerLoopB16_);
        }

        // 尾loop
        if (blockIdx_ == handleExtraLoopCoreNumB16_ - 1) {
            CopyIn(loopNumPerCoreB16_, numLastLoopB16_);
            Compute(numLastLoopB16_);
            CopyOut(loopNumPerCoreB16_, numLastLoopB16_);
        }
    }
}

} // namespace ApplyAdamWV2

#endif // APPLYADAM_W_V2_B16_H
