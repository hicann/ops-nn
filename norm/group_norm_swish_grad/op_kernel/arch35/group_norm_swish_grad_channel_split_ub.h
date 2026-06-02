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
 * \file group_norm_swish_grad_channel_split_ub.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_GRAD_CHANNEL_SPLIT_UB_H
#define GROUP_NORM_SWISH_GRAD_CHANNEL_SPLIT_UB_H

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitChannelSplitUbBuffer(
    GroupNormSwishGradTilingData* tilingData)
{
    this->channelTileCapacityEle = tilingData->mode2_ub_capacity_ele;
    this->channelTileIterationNum = tilingData->mode2_ub_iteration_num;
    this->channelTileTailNum = tilingData->mode2_ub_tail_num;
    uint32_t meanRstdAllocSpace = meanRstdAlignedFloatEleCount * sizeof(float);
    uint32_t meanRstdTAllocSpace = meanRstdTempEleCount * sizeof(T);
    pipe.InitBuffer(inTbufMeanRstd, meanRstdAllocSpace);
    if constexpr (IsSameType<T, float>::value) {
        uint32_t capacityEleAllocSpace = Ceil(this->channelTileCapacityEle, tPerBlock) * sizeof(float);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock) * sizeof(float);
        pipe.InitBuffer(inQueueDy, 1, capacityEleAllocSpace);
        pipe.InitBuffer(inQueueX, 1, capacityEleAllocSpace);
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace);
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace);
        pipe.InitBuffer(outTbufDyNew, capacityEleAllocSpace);
        pipe.InitBuffer(outTbufDswish, capacityEleAllocSpace);
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace);
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace);
        pipe.InitBuffer(tempQueueHxw, 1, capacityEleAllocSpace);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        uint32_t capacityEleAllocSpace = Ceil(this->channelTileCapacityEle, tPerBlock);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock);
        pipe.InitBuffer(inQueueMeanRstdT, 1, meanRstdTAllocSpace);
        pipe.InitBuffer(inQueueDy, 1, capacityEleAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueDyT, 1, capacityEleAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueX, 1, capacityEleAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueXT, 1, capacityEleAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueGammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueBetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDxT, 1, capacityEleAllocSpace * sizeof(T));
        pipe.InitBuffer(outTbufDyNew, capacityEleAllocSpace * sizeof(float));
        pipe.InitBuffer(outTbufDswish, capacityEleAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDgammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDbetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(tempQueueHxw, 1, capacityEleAllocSpace * sizeof(float));
    }
}

// One channel HxW cannot fit in UB, so copy one channel tile at a time.
// channelTile* fields describe tile capacity, tile count, and tail size.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CopyInChannelSplitUb(
    uint64_t offset, uint32_t processNum, const LocalTensor<float>& meanRstdLocal)
{
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> dyLocal = inQueueDy.AllocTensor<float>();
    CustomDataCopyIn(xLocal, inQueueXT, xGm, offset, processNum);
    CustomDataCopyIn(dyLocal, inQueueDyT, dyGm, offset, processNum);
    MulsAddsTemplate(xLocal, meanRstdLocal, processNum);
    inQueueX.EnQue(xLocal);
    inQueueDy.EnQue(dyLocal);
}

// Partial dgamma/dbeta reduction pass for one channel tile.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ComputeDgammaDbetaChannelSplitUb(
    int32_t iterCGIdx, const LocalTensor<float>& temp1Local, const LocalTensor<float>& temp2Local,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, uint64_t offset, uint32_t processNum)
{
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> dyLocal = inQueueDy.DeQue<float>();
    LocalTensor<float> dyNew = outTbufDyNew.Get<float>(Ceil(this->channelTileCapacityEle, tPerBlock));
    LocalTensor<float> dswishRes = outTbufDswish.Get<float>(Ceil(this->channelTileCapacityEle, tPerBlock));

    float swishScaleValue = this->swishScale;
    uint32_t cGIdx = static_cast<uint32_t>(iterCGIdx / this->channelTileIterationNum);
    SwishGradMulXReduceTemplate(
        temp1Local, temp2Local, dyNew, dswishRes, xLocal, dyLocal, gammaLocal, betaLocal, cGIdx, swishScaleValue,
        processNum);
    inQueueDy.FreeTensor(dyLocal);
    inQueueX.FreeTensor(xLocal);
}

// Channel-split path: HxW does not fit as a whole, so each channel is split into UB-sized
// segments. Partial reductions accumulate into temp1/temp2 by channel index.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ComputeChannelSplitUb(int32_t taskIdx)
{
    LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
    LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();
    LocalTensor<float> temp1Local = outQueueDbeta.AllocTensor<float>();
    LocalTensor<float> temp2Local = outQueueDgamma.AllocTensor<float>();
    LocalTensor<float> meanRstdLocal = inTbufMeanRstd.Get<float>(meanRstdAlignedFloatEleCount);
    uint32_t channelIdx = (taskIdx % this->g) * this->cG;
    Duplicate(temp1Local, 0.0f, this->cG);
    Duplicate(temp2Local, 0.0f, this->cG);
    PipeBarrier<PIPE_V>();
    CopyMeanRstdToUb(meanRstdLocal, taskIdx);
    CustomDataCopyIn(gammaLocal, inQueueGammaT, gammaGm, channelIdx, this->cG);
    CustomDataCopyIn(betaLocal, inQueueBetaT, betaGm, channelIdx, this->cG);
    for (int32_t iterCGIdx = 0; iterCGIdx < this->cG * this->channelTileIterationNum; iterCGIdx++) {
        uint64_t offset =
            static_cast<uint64_t>(taskIdx) * this->eleNumPerGroup +
            static_cast<uint64_t>(iterCGIdx / this->channelTileIterationNum) * this->eleNumPerChannel +
            static_cast<uint64_t>(iterCGIdx % this->channelTileIterationNum) * this->channelTileCapacityEle;
        if (iterCGIdx % this->channelTileIterationNum != this->channelTileIterationNum - 1) {
            CopyInChannelSplitUb(offset, this->channelTileCapacityEle, meanRstdLocal);
            ComputeDgammaDbetaChannelSplitUb(
                iterCGIdx, temp1Local, temp2Local, gammaLocal, betaLocal, offset, this->channelTileCapacityEle);
        } else {
            CopyInChannelSplitUb(offset, this->channelTileTailNum, meanRstdLocal);
            ComputeDgammaDbetaChannelSplitUb(
                iterCGIdx, temp1Local, temp2Local, gammaLocal, betaLocal, offset, this->channelTileTailNum);
        }
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    ModeSelection(taskIdx, temp2Local, temp1Local);
    float swishScaleValue = this->swishScale;
    MulReduceSumTemplate(temp1Local, temp2Local, gammaLocal, cG);
    for (int32_t iterCGIdx = 0; iterCGIdx < this->cG * this->channelTileIterationNum; iterCGIdx++) {
        uint64_t offset =
            static_cast<uint64_t>(taskIdx) * this->eleNumPerGroup +
            static_cast<uint64_t>(iterCGIdx / this->channelTileIterationNum) * this->eleNumPerChannel +
            static_cast<uint64_t>(iterCGIdx % this->channelTileIterationNum) * this->channelTileCapacityEle;
        uint32_t cGIdx = static_cast<uint32_t>(iterCGIdx / this->channelTileIterationNum);
        if (iterCGIdx % this->channelTileIterationNum != this->channelTileIterationNum - 1) {
            LocalTensor<float> tempHxwLocal = tempQueueHxw.AllocTensor<float>();
            CopyInChannelSplitUb(offset, this->channelTileCapacityEle, meanRstdLocal);
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> dyLocal = inQueueDy.DeQue<float>();
            SwishDxTemplate(
                tempHxwLocal, xLocal, dyLocal, gammaLocal, betaLocal, temp1Local, temp2Local, meanRstdLocal, cGIdx,
                swishScaleValue, this->channelTileCapacityEle);
            inQueueX.FreeTensor(xLocal);
            inQueueDy.FreeTensor(dyLocal);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CustomDataCopyOut(tempHxwLocal, offset, outQueueDxT, this->channelTileCapacityEle);
            tempQueueHxw.FreeTensor(tempHxwLocal);
        } else {
            LocalTensor<float> tempHxwLocal = tempQueueHxw.AllocTensor<float>();
            CopyInChannelSplitUb(offset, this->channelTileTailNum, meanRstdLocal);
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> dyLocal = inQueueDy.DeQue<float>();
            SwishDxTemplate(
                tempHxwLocal, xLocal, dyLocal, gammaLocal, betaLocal, temp1Local, temp2Local, meanRstdLocal, cGIdx,
                swishScaleValue, this->channelTileTailNum);
            inQueueX.FreeTensor(xLocal);
            inQueueDy.FreeTensor(dyLocal);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CustomDataCopyOut(tempHxwLocal, offset, outQueueDxT, this->channelTileTailNum);
            tempQueueHxw.FreeTensor(tempHxwLocal);
        }
    }
    outQueueDbeta.FreeTensor(temp1Local);
    outQueueDgamma.FreeTensor(temp2Local);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
}

#endif // GROUP_NORM_SWISH_GRAD_CHANNEL_SPLIT_UB_H
