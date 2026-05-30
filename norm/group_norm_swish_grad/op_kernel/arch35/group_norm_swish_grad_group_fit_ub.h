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
 * \file group_norm_swish_grad_group_fit_ub.h
 * \brief
 */

#ifndef GROUP_NORM_SWISH_GRAD_GROUP_FIT_UB_H
#define GROUP_NORM_SWISH_GRAD_GROUP_FIT_UB_H

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::InitGroupFitUbBuffer()
{
    uint32_t meanRstdAllocSpace = meanRstdAlignedFloatEleCount * sizeof(float);
    uint32_t meanRstdTAllocSpace = meanRstdTempEleCount * sizeof(T);
    pipe.InitBuffer(inTbufMeanRstd, meanRstdAllocSpace);
    if constexpr (IsSameType<T, float>::value) {
        uint32_t hxwAllocSpace = Ceil(this->eleNumPerChannel, tPerBlock) * sizeof(T);
        uint32_t cGxHxwAllocSpace = Ceil(this->eleNumPerGroup, tPerBlock) * sizeof(T);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock) * sizeof(T);
        pipe.InitBuffer(inQueueDy, 1, cGxHxwAllocSpace);
        pipe.InitBuffer(inQueueX, 1, cGxHxwAllocSpace);
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace);
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace);
        pipe.InitBuffer(outTbufDyNew, hxwAllocSpace);
        pipe.InitBuffer(outTbufDswish, hxwAllocSpace);
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace);
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace);
        pipe.InitBuffer(tempQueueHxw, 1, cGxHxwAllocSpace);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        uint32_t hxwAllocSpace = Ceil(this->eleNumPerChannel, tPerBlock);
        uint32_t cGxHxwAllocSpace = Ceil(this->eleNumPerGroup, tPerBlock);
        uint32_t cGAllocSpace = Ceil(this->cG, tPerBlock);
        pipe.InitBuffer(inQueueMeanRstdT, 1, meanRstdTAllocSpace);
        pipe.InitBuffer(inQueueDy, 1, cGxHxwAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueDyT, 1, cGxHxwAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueX, 1, cGxHxwAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueXT, 1, cGxHxwAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueGamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueGammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(inQueueBeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(inQueueBetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDxT, 1, hxwAllocSpace * sizeof(T));
        pipe.InitBuffer(outTbufDyNew, cGxHxwAllocSpace * sizeof(float));
        pipe.InitBuffer(outTbufDswish, cGxHxwAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDgammaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDbetaT, 1, cGAllocSpace * sizeof(T));
        pipe.InitBuffer(outQueueDgamma, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(outQueueDbeta, 1, cGAllocSpace * sizeof(float));
        pipe.InitBuffer(tempQueueHxw, 1, cGxHxwAllocSpace * sizeof(float));
    }
}

// UB can hold one whole group (C/G * HxW). Copy the group once,
// normalize x once, then walk channels inside the resident UB block.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::CopyInGroupFitUb(int32_t taskIdx)
{
    uint64_t offset = static_cast<uint64_t>(taskIdx) * this->eleNumPerGroup;
    uint32_t onceProcessEleNum = static_cast<uint32_t>(this->eleNumPerGroup);
    LocalTensor<float> dyLocal = inQueueDy.AllocTensor<float>();
    CustomDataCopyIn(dyLocal, inQueueDyT, dyGm, offset, onceProcessEleNum);
    inQueueDy.EnQue(dyLocal);
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    CustomDataCopyIn(xLocal, inQueueXT, xGm, offset, onceProcessEleNum);
    LocalTensor<float> meanRstdLocal = inTbufMeanRstd.Get<float>(meanRstdAlignedFloatEleCount);
    CopyMeanRstdToUb(meanRstdLocal, taskIdx);
    MulsAddsTemplate(xLocal, meanRstdLocal, this->eleNumPerGroup);
    inQueueX.EnQue(xLocal);
}

// Whole-group-in-UB compute path: gamma/beta are loaded once. Channels with
// unaligned UB starts use the unaligned VF load path directly, without GM recopy.
template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::ComputeGroupFitUb(int32_t taskIdx)
{
    uint32_t channelIdx = (taskIdx % this->g) * this->cG;
    LocalTensor<float> dyLocal = inQueueDy.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
    LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();
    LocalTensor<float> temp1Local = outQueueDbeta.AllocTensor<float>();
    LocalTensor<float> temp2Local = outQueueDgamma.AllocTensor<float>();
    LocalTensor<float> tempHxwLocal = tempQueueHxw.AllocTensor<float>();
    LocalTensor<float> meanRstdLocal = inTbufMeanRstd.Get<float>(meanRstdAlignedFloatEleCount);
    LocalTensor<float> dyNew = outTbufDyNew.Get<float>(this->eleNumPerChannel);
    LocalTensor<float> dswishRes = outTbufDswish.Get<float>(this->eleNumPerChannel);
    Duplicate(temp1Local, 0.0f, this->cG);
    Duplicate(temp2Local, 0.0f, this->cG);
    PipeBarrier<PIPE_V>();

    CustomDataCopyIn(gammaLocal, inQueueGammaT, gammaGm, channelIdx, this->cG);
    CustomDataCopyIn(betaLocal, inQueueBetaT, betaGm, channelIdx, this->cG);
    float swishScaleValue = this->swishScale;
    for (int32_t cGIdx = 0; cGIdx < this->cG; cGIdx++) {
        int32_t isAlign = (cGIdx * this->eleNumPerChannel) % this->floatPerBlock;
        if (isAlign == 0) {
            uint32_t offset = cGIdx * this->eleNumPerChannel;
            SwishGradMulXReduceTemplate(temp1Local, temp2Local, dyNew, dswishRes, xLocal[offset], dyLocal[offset],
                gammaLocal, betaLocal, static_cast<uint32_t>(cGIdx), swishScaleValue, this->eleNumPerChannel);
        } else {
            uint32_t offset = cGIdx * this->eleNumPerChannel;
            SwishGradMulXReduceUnAlignTemplate(temp1Local, temp2Local, dyNew, dswishRes, xLocal[offset],
                dyLocal[offset], gammaLocal, betaLocal, static_cast<uint32_t>(cGIdx), swishScaleValue,
                this->eleNumPerChannel);
        }
    }
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    ModeSelection(taskIdx, temp2Local, temp1Local);
    MulReduceSumTemplate(temp1Local, temp2Local, gammaLocal, cG);
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    for (int32_t cGIdx = 0; cGIdx < this->cG; cGIdx++) {
        int32_t isAlign = (cGIdx * this->eleNumPerChannel) % this->tPerBlock;
        uint64_t gmOffset = static_cast<uint64_t>(taskIdx) * this->eleNumPerGroup + static_cast<uint64_t>(cGIdx) * this->eleNumPerChannel;
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        if (isAlign == 0) {
            uint32_t offset = cGIdx * this->eleNumPerChannel;
            SwishDxTemplate(tempHxwLocal, xLocal[offset], dyLocal[offset], gammaLocal, betaLocal, temp1Local,
                temp2Local, meanRstdLocal, static_cast<uint32_t>(cGIdx), swishScaleValue, this->eleNumPerChannel);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CustomDataCopyOut(tempHxwLocal, gmOffset, outQueueDxT, eleNumPerChannel);
        } else {
            uint32_t offset = cGIdx * this->eleNumPerChannel;
            SwishDxUnAlignTemplate(tempHxwLocal, xLocal[offset], dyLocal[offset], gammaLocal, betaLocal, temp1Local,
                temp2Local, meanRstdLocal, static_cast<uint32_t>(cGIdx), swishScaleValue, this->eleNumPerChannel);
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            CustomDataCopyOut(tempHxwLocal, gmOffset, outQueueDxT, eleNumPerChannel);
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    tempQueueHxw.FreeTensor(tempHxwLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueueDbeta.FreeTensor(temp1Local);
    outQueueDgamma.FreeTensor(temp2Local);
    inQueueDy.FreeTensor(dyLocal);
}

// Group-fit-only unaligned UB helpers.

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishGradMulXReduceUnAlignTemplate(
    const LocalTensor<float>& temp1Local, const LocalTensor<float>& temp2Local,
    const LocalTensor<float>& dyNewSumLocal, const LocalTensor<float>& xMulDySumLocal,
    const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal, const LocalTensor<float>& gammaLocal,
    const LocalTensor<float>& betaLocal, uint32_t cGIdx, float swishScaleValue, uint32_t calCount)
{
    __ubuf__ float* temp1Ptr = (__ubuf__ float*)temp1Local.GetPhyAddr();
    __ubuf__ float* temp2Ptr = (__ubuf__ float*)temp2Local.GetPhyAddr();
    __ubuf__ float* dyNewSumPtr = (__ubuf__ float*)dyNewSumLocal.GetPhyAddr();
    __ubuf__ float* xMulDySumPtr = (__ubuf__ float*)xMulDySumLocal.GetPhyAddr();
    __ubuf__ float* xPtr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* dyPtr = (__ubuf__ float*)dyLocal.GetPhyAddr();
    __ubuf__ float* gammaPtr = (__ubuf__ float*)gammaLocal.GetPhyAddr();
    __ubuf__ float* betaPtr = (__ubuf__ float*)betaLocal.GetPhyAddr();
    PipeBarrier<PIPE_V>();
    SwishGradMulXReduceUnAlignVf(temp1Ptr, temp2Ptr, dyNewSumPtr, xMulDySumPtr, xPtr, dyPtr, gammaPtr, betaPtr,
        cGIdx, swishScaleValue, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isDeterministic>
__aicore__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishDxUnAlignTemplate(
    const LocalTensor<float>& dxLocal, const LocalTensor<float>& xLocal, const LocalTensor<float>& dyLocal,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal, const LocalTensor<float>& c1Local,
    const LocalTensor<float>& c2Local, const LocalTensor<float>& meanRstdLocal, uint32_t cGIdx,
    float swishScaleValue,
    uint32_t calCount)
{
    __ubuf__ float* dxPtr = (__ubuf__ float*)dxLocal.GetPhyAddr();
    __ubuf__ float* xPtr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* dyPtr = (__ubuf__ float*)dyLocal.GetPhyAddr();
    __ubuf__ float* gammaPtr = (__ubuf__ float*)gammaLocal.GetPhyAddr();
    __ubuf__ float* betaPtr = (__ubuf__ float*)betaLocal.GetPhyAddr();
    __ubuf__ float* c1Ptr = (__ubuf__ float*)c1Local.GetPhyAddr();
    __ubuf__ float* c2Ptr = (__ubuf__ float*)c2Local.GetPhyAddr();
    __ubuf__ float* meanRstdPtr = (__ubuf__ float*)meanRstdLocal.GetPhyAddr();
    PipeBarrier<PIPE_V>();
    SwishDxUnAlignVf(
        dxPtr, xPtr, dyPtr, gammaPtr, betaPtr, c1Ptr, c2Ptr, meanRstdPtr, cGIdx, swishScaleValue, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishDxUnAlignVf(__ubuf__ float* dxPtr,
                                                                                 __ubuf__ float* xPtr,
                                                                                 __ubuf__ float* dyPtr,
                                                                                 __ubuf__ float* gammaPtr,
                                                                                 __ubuf__ float* betaPtr,
                                                                                 __ubuf__ float* c1Ptr,
                                                                                 __ubuf__ float* c2Ptr,
                                                                                 __ubuf__ float* meanRstdPtr,
                                                                                 uint32_t cGIdx,
                                                                                 float swishScaleValue,
                                                                                 uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskFull0;
    AscendC::MicroAPI::MaskReg maskFull1;
    AscendC::MicroAPI::UnalignRegForLoad uLoadX;
    AscendC::MicroAPI::UnalignRegForLoad uLoadDy;
    AscendC::MicroAPI::RegTensor<float> vGamma, vBeta, vC1, vC2, vRstd;
    AscendC::MicroAPI::RegTensor<float> vX0, vDy0, vYMul0, vY0, vDenScaled0, vDenExp0, vDen0;
    AscendC::MicroAPI::RegTensor<float> vDswish0, vDyNew0, vCorrectionMul0, vCorrection0, vDx0;
    AscendC::MicroAPI::RegTensor<float> vX1, vDy1, vYMul1, vY1, vDenScaled1, vDenExp1, vDen1;
    AscendC::MicroAPI::RegTensor<float> vDswish1, vDyNew1, vCorrectionMul1, vCorrection1, vDx1;
    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) / 2);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vGamma, gammaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vBeta, betaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vC1, c1Ptr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vC2, c2Ptr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
        vRstd, meanRstdPtr + rstdValueOffset);
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        uint32_t offset0 = i * 2 * oneRepeatSizeB32;
        uint32_t offset1 = offset0 + oneRepeatSizeB32;
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadX, xPtr + offset0);
        AscendC::MicroAPI::LoadUnAlign(vX0, uLoadX, xPtr + offset0);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadDy, dyPtr + offset0);
        AscendC::MicroAPI::LoadUnAlign(vDy0, uLoadDy, dyPtr + offset0);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadX, xPtr + offset1);
        AscendC::MicroAPI::LoadUnAlign(vX1, uLoadX, xPtr + offset1);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadDy, dyPtr + offset1);
        AscendC::MicroAPI::LoadUnAlign(vDy1, uLoadDy, dyPtr + offset1);
        AscendC::MicroAPI::Mul(vYMul0, vX0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vYMul1, vX1, vGamma, maskFull1);
        AscendC::MicroAPI::Add(vY0, vYMul0, vBeta, maskFull0);
        AscendC::MicroAPI::Add(vY1, vYMul1, vBeta, maskFull1);
        AscendC::MicroAPI::Muls(vYMul0, vY0, swishScaleValue, maskFull0);
        AscendC::MicroAPI::Muls(vYMul1, vY1, swishScaleValue, maskFull1);
        AscendC::MicroAPI::Muls(vDenScaled0, vYMul0, float(-1.0), maskFull0);
        AscendC::MicroAPI::Muls(vDenScaled1, vYMul1, float(-1.0), maskFull1);
        AscendC::MicroAPI::Exp(vDenExp0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Exp(vDenExp1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vDen0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vDen1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDenScaled0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDenScaled1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Sub(vDenExp0, vYMul0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Sub(vDenExp1, vYMul1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vYMul0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vYMul1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDswish0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDswish1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vDy0, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vDy1, maskFull1);
        AscendC::MicroAPI::Mul(vDswish0, vDyNew0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vDswish1, vDyNew1, vGamma, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vRstd, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vRstd, maskFull1);
        AscendC::MicroAPI::Mul(vCorrectionMul0, vX0, vC2, maskFull0);
        AscendC::MicroAPI::Mul(vCorrectionMul1, vX1, vC2, maskFull1);
        AscendC::MicroAPI::Add(vCorrection0, vCorrectionMul0, vC1, maskFull0);
        AscendC::MicroAPI::Add(vCorrection1, vCorrectionMul1, vC1, maskFull1);
        AscendC::MicroAPI::Mul(vCorrectionMul0, vCorrection0, vRstd, maskFull0);
        AscendC::MicroAPI::Mul(vCorrectionMul1, vCorrection1, vRstd, maskFull1);
        AscendC::MicroAPI::Sub(vDx0, vDyNew0, vCorrectionMul0, maskFull0);
        AscendC::MicroAPI::Sub(vDx1, vDyNew1, vCorrectionMul1, maskFull1);
        AscendC::MicroAPI::StoreAlign(dxPtr + offset0, vDx0, maskFull0);
        AscendC::MicroAPI::StoreAlign(dxPtr + offset1, vDx1, maskFull1);
    }
}

template <typename T, bool isDeterministic>
__simd_vf__ inline void GroupNormSwishGrad<T, isDeterministic>::SwishGradMulXReduceUnAlignVf(
    __ubuf__ float* temp1Ptr, __ubuf__ float* temp2Ptr, __ubuf__ float* dyNewSumPtr,
    __ubuf__ float* xMulDySumPtr, __ubuf__ float* xPtr, __ubuf__ float* dyPtr, __ubuf__ float* gammaPtr,
    __ubuf__ float* betaPtr, uint32_t cGIdx, float swishScaleValue, uint32_t calCount)
{
    AscendC::MicroAPI::MaskReg maskFull0;
    AscendC::MicroAPI::MaskReg maskFull1;
    AscendC::MicroAPI::MaskReg maskScalar;
    AscendC::MicroAPI::UnalignRegForLoad uLoadX;
    AscendC::MicroAPI::UnalignRegForLoad uLoadDy;
    AscendC::MicroAPI::UnalignRegForStore uStoreDyNewSum;
    AscendC::MicroAPI::UnalignRegForStore uStoreXMulDySum;
    AscendC::MicroAPI::UnalignRegForStore uStoreTemp1;
    AscendC::MicroAPI::UnalignRegForStore uStoreTemp2;
    AscendC::MicroAPI::RegTensor<float> vGamma, vBeta;
    AscendC::MicroAPI::RegTensor<float> vX0, vDy0, vYMul0, vY0, vDenScaled0, vDenExp0, vDen0;
    AscendC::MicroAPI::RegTensor<float> vDswish0, vDyNew0, vXMulDy0, vTemp1Acc;
    AscendC::MicroAPI::RegTensor<float> vX1, vDy1, vYMul1, vY1, vDenScaled1, vDenExp1, vDen1;
    AscendC::MicroAPI::RegTensor<float> vDswish1, vDyNew1, vXMulDy1, vTemp2Acc;
    uint32_t storeCount = 1;
    maskScalar = AscendC::MicroAPI::UpdateMask<float>(storeCount);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vGamma, gammaPtr + cGIdx);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vBeta, betaPtr + cGIdx);

    uint32_t fullRepeatTimes = (calCount + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    uint16_t pairRepeatTimes = static_cast<uint16_t>((fullRepeatTimes + 1) / 2);
    __ubuf__ float* dyNewSumStorePtr = dyNewSumPtr;
    __ubuf__ float* xMulDySumStorePtr = xMulDySumPtr;
    // Unaligned source path mirrors the aligned path. Pre/Load must be paired
    // for each source address, with separate unalign registers for x and dy.
    for (uint16_t i = 0; i < pairRepeatTimes; i++) {
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(calCount);
        uint32_t offset0 = i * 2 * oneRepeatSizeB32;
        uint32_t offset1 = offset0 + oneRepeatSizeB32;
        AscendC::MicroAPI::LoadUnAlignPre(uLoadX, xPtr + offset0);
        AscendC::MicroAPI::LoadUnAlign(vX0, uLoadX, xPtr + offset0);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadDy, dyPtr + offset0);
        AscendC::MicroAPI::LoadUnAlign(vDy0, uLoadDy, dyPtr + offset0);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadX, xPtr + offset1);
        AscendC::MicroAPI::LoadUnAlign(vX1, uLoadX, xPtr + offset1);
        AscendC::MicroAPI::LoadUnAlignPre(uLoadDy, dyPtr + offset1);
        AscendC::MicroAPI::LoadUnAlign(vDy1, uLoadDy, dyPtr + offset1);
        AscendC::MicroAPI::Mul(vYMul0, vX0, vGamma, maskFull0);
        AscendC::MicroAPI::Mul(vYMul1, vX1, vGamma, maskFull1);
        AscendC::MicroAPI::Add(vY0, vYMul0, vBeta, maskFull0);
        AscendC::MicroAPI::Add(vY1, vYMul1, vBeta, maskFull1);
        AscendC::MicroAPI::Muls(vYMul0, vY0, swishScaleValue, maskFull0);
        AscendC::MicroAPI::Muls(vYMul1, vY1, swishScaleValue, maskFull1);
        AscendC::MicroAPI::Muls(vDenScaled0, vYMul0, float(-1.0), maskFull0);
        AscendC::MicroAPI::Muls(vDenScaled1, vYMul1, float(-1.0), maskFull1);
        AscendC::MicroAPI::Exp(vDenExp0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Exp(vDenExp1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vDen0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vDen1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDenScaled0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDenScaled1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Sub(vDenExp0, vYMul0, vDenScaled0, maskFull0);
        AscendC::MicroAPI::Sub(vDenExp1, vYMul1, vDenScaled1, maskFull1);
        AscendC::MicroAPI::Adds(vYMul0, vDenExp0, float(1.0), maskFull0);
        AscendC::MicroAPI::Adds(vYMul1, vDenExp1, float(1.0), maskFull1);
        AscendC::MicroAPI::Div(vDswish0, vYMul0, vDen0, maskFull0);
        AscendC::MicroAPI::Div(vDswish1, vYMul1, vDen1, maskFull1);
        AscendC::MicroAPI::Mul(vDyNew0, vDswish0, vDy0, maskFull0);
        AscendC::MicroAPI::Mul(vDyNew1, vDswish1, vDy1, maskFull1);
        AscendC::MicroAPI::Mul(vXMulDy0, vX0, vDyNew0, maskFull0);
        AscendC::MicroAPI::Mul(vXMulDy1, vX1, vDyNew1, maskFull1);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vYMul1, vDyNew1, maskFull1);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled1, vXMulDy1, maskFull1);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul1, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled1, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));
    uint32_t sumReduceCount = fullRepeatTimes;
    uint32_t sumRepeatTimes = (fullRepeatTimes + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    dyNewSumStorePtr = dyNewSumPtr;
    xMulDySumStorePtr = xMulDySumPtr;
    for (uint16_t i = 0; i < sumRepeatTimes; i++) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(sumReduceCount);
        uint32_t offset = i * oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr + offset);
        AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr + offset);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull0);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));

    uint32_t finalReduceCount0 = sumRepeatTimes;
    uint32_t finalReduceCount1 = sumRepeatTimes;
    uint32_t finalRepeatTimes = (sumRepeatTimes + oneRepeatSizeB32 - 1) / oneRepeatSizeB32;
    dyNewSumStorePtr = dyNewSumPtr;
    xMulDySumStorePtr = xMulDySumPtr;
    for (uint16_t i = 0; i < finalRepeatTimes; i++) {
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        maskFull0 = AscendC::MicroAPI::UpdateMask<float>(finalReduceCount0);
        maskFull1 = AscendC::MicroAPI::UpdateMask<float>(finalReduceCount1);
        uint32_t offset = i * oneRepeatSizeB32;
        AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr + offset);
        AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr + offset);
        AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
        AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull1);
        AscendC::MicroAPI::StoreUnAlign(dyNewSumStorePtr, vYMul0, uStoreDyNewSum, static_cast<uint32_t>(1));
        AscendC::MicroAPI::StoreUnAlign(xMulDySumStorePtr, vDenScaled0, uStoreXMulDySum, static_cast<uint32_t>(1));
    }
    AscendC::MicroAPI::StoreUnAlignPost(dyNewSumStorePtr, uStoreDyNewSum, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlignPost(xMulDySumStorePtr, uStoreXMulDySum, static_cast<int32_t>(0));

    uint32_t lastReduceCount0 = finalRepeatTimes;
    uint32_t lastReduceCount1 = finalRepeatTimes;
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    maskFull0 = AscendC::MicroAPI::UpdateMask<float>(lastReduceCount0);
    maskFull1 = AscendC::MicroAPI::UpdateMask<float>(lastReduceCount1);
    AscendC::MicroAPI::LoadAlign(vDyNew0, dyNewSumPtr);
    AscendC::MicroAPI::LoadAlign(vXMulDy0, xMulDySumPtr);
    AscendC::MicroAPI::ReduceSum(vYMul0, vDyNew0, maskFull0);
    AscendC::MicroAPI::ReduceSum(vDenScaled0, vXMulDy0, maskFull1);
    __ubuf__ float* temp1LoadPtr = temp1Ptr + cGIdx;
    __ubuf__ float* temp2LoadPtr = temp2Ptr + cGIdx;
    __ubuf__ float* temp1StorePtr = temp1Ptr + cGIdx;
    __ubuf__ float* temp2StorePtr = temp2Ptr + cGIdx;
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vTemp1Acc, temp1LoadPtr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vTemp2Acc, temp2LoadPtr);
    AscendC::MicroAPI::Add(vYMul1, vYMul0, vTemp1Acc, maskScalar);
    AscendC::MicroAPI::Add(vDenScaled1, vDenScaled0, vTemp2Acc, maskScalar);
    AscendC::MicroAPI::StoreUnAlign(temp1StorePtr, vYMul1, uStoreTemp1, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlignPost(temp1StorePtr, uStoreTemp1, static_cast<int32_t>(0));
    AscendC::MicroAPI::StoreUnAlign(temp2StorePtr, vDenScaled1, uStoreTemp2, static_cast<uint32_t>(1));
    AscendC::MicroAPI::StoreUnAlignPost(temp2StorePtr, uStoreTemp2, static_cast<int32_t>(0));
}

#endif // GROUP_NORM_SWISH_GRAD_GROUP_FIT_UB_H
