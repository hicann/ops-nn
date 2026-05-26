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
 * \file group_norm_swish_grad_channel_fit_ub.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_GRAD_CHANNEL_FIT_UB_H
#define GROUP_NORM_SWISH_GRAD_CHANNEL_FIT_UB_H

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitChannelFitUbBuffer()
{
    uint32_t meanRstdAllocSpace = meanRstdAlignedFloatEleCount * sizeof(float);
    uint32_t meanRstdTAllocSpace = meanRstdTempEleCount * sizeof(T);
    pipe.InitBuffer(inTbufMeanRstd, meanRstdAllocSpace);
    if constexpr (IsSameType<T, float>::value) {
        uint32_t hxwAllocSpace = Ceil(this->eleNumPerChannel, tPerBlock) * sizeof(float);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock) * sizeof(float);
        pipe.InitBuffer(inQueueDy, 1, hxwAllocSpace);
        pipe.InitBuffer(inQueueX, 1, hxwAllocSpace);
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace);
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace);
        pipe.InitBuffer(outTbufDyNew, hxwAllocSpace);
        pipe.InitBuffer(outTbufDswish, hxwAllocSpace);
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace);
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace);
        pipe.InitBuffer(tempQueueHxw, 1, hxwAllocSpace);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        uint32_t hxwAllocSpace = Ceil(this->eleNumPerChannel, tPerBlock);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock);
        pipe.InitBuffer(inQueueMeanRstdT, 1, meanRstdTAllocSpace);
        pipe.InitBuffer(inQueueDy, 1, hxwAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueDyT, 1, hxwAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueX, 1, hxwAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueXT, 1, hxwAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueGammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueBetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDxT, 1, hxwAllocSpace * sizeof(T));
        pipe.InitBuffer(outTbufDyNew, hxwAllocSpace * sizeof(float));
        pipe.InitBuffer(outTbufDswish, hxwAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDgammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDbetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(tempQueueHxw, 1, hxwAllocSpace * sizeof(float));
    }
}

// UB can hold one channel HxW, while the full group does not.
// Each channel is copied, normalized, reduced, and later revisited for dx.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ComputeChannelFitUb(int32_t taskIdx)
{
    LocalTensor<float> temp1Local = outQueueDbeta.AllocTensor<float>();
    LocalTensor<float> temp2Local = outQueueDgamma.AllocTensor<float>();
    uint32_t baseOffset = taskIdx * this->eleNumPerGroup;
    uint32_t onceProcessEleNum = this->eleNumPerChannel;
    uint32_t channelIdx = (taskIdx % this->g) * this->cG;

    LocalTensor<float> dyLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> tempHxwLocal;
    LocalTensor<float> meanRstdLocal = inTbufMeanRstd.Get<float>(meanRstdAlignedFloatEleCount);
    LocalTensor<float> dyNew = outTbufDyNew.Get<float>(this->eleNumPerChannel);
    LocalTensor<float> dswishRes = outTbufDswish.Get<float>(this->eleNumPerChannel);
    LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
    LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();
    Duplicate(temp1Local, 0.0f, this->cG);
    Duplicate(temp2Local, 0.0f, this->cG);
    PipeBarrier<PIPE_V>();
    CopyMeanRstdToUb(meanRstdLocal, taskIdx);
    CustomDataCopyIn(gammaLocal, inQueueGammaT, gammaGm, channelIdx, this->cG);
    CustomDataCopyIn(betaLocal, inQueueBetaT, betaGm, channelIdx, this->cG);
    float swishScaleValue = this->swishScale;
    for (int32_t cGIdx = 0; cGIdx < this->cG; cGIdx++) {
        uint32_t offset = baseOffset + cGIdx * this->eleNumPerChannel;
        xLocal = inQueueX.AllocTensor<float>();
        dyLocal = inQueueDy.AllocTensor<float>();
        CustomDataCopyIn(xLocal, inQueueXT, xGm, offset, onceProcessEleNum);
        CustomDataCopyIn(dyLocal, inQueueDyT, dyGm, offset, onceProcessEleNum);
        MulsAddsTemplate(xLocal, meanRstdLocal, this->eleNumPerChannel);
        SwishGradMulXReduceTemplate(temp1Local, temp2Local, dyNew, dswishRes, xLocal, dyLocal,
            gammaLocal, betaLocal, static_cast<uint32_t>(cGIdx), swishScaleValue, this->eleNumPerChannel);
        inQueueX.FreeTensor(xLocal);
        inQueueDy.FreeTensor(dyLocal);
    }
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    ModeSelection(taskIdx, temp2Local, temp1Local);
    MulReduceSumTemplate(temp1Local, temp2Local, gammaLocal, cG);
    for (int32_t cGIdx = 0; cGIdx < this->cG; cGIdx++) {
        uint32_t offset = baseOffset + cGIdx * this->eleNumPerChannel;
        xLocal = inQueueX.AllocTensor<float>();
        dyLocal = inQueueDy.AllocTensor<float>();
        tempHxwLocal = tempQueueHxw.AllocTensor<float>();
        CustomDataCopyIn(xLocal, inQueueXT, xGm, offset, onceProcessEleNum);
        CustomDataCopyIn(dyLocal, inQueueDyT, dyGm, offset, onceProcessEleNum);
        MulsAddsTemplate(xLocal, meanRstdLocal, this->eleNumPerChannel);
        SwishDxTemplate(tempHxwLocal, xLocal, dyLocal, gammaLocal, betaLocal, temp1Local, temp2Local,
            meanRstdLocal, static_cast<uint32_t>(cGIdx), swishScaleValue, this->eleNumPerChannel);
        inQueueX.FreeTensor(xLocal);
        inQueueDy.FreeTensor(dyLocal);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        CustomDataCopyOut(tempHxwLocal, offset, outQueueDxT, eleNumPerChannel);
        tempQueueHxw.FreeTensor(tempHxwLocal);
    }
    outQueueDbeta.FreeTensor(temp1Local);
    outQueueDgamma.FreeTensor(temp2Local);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
}

#endif // GROUP_NORM_SWISH_GRAD_CHANNEL_FIT_UB_H
