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
 * \file avg_pool_v2_grad_nhwc_kernel.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_NHWC_KERNEL_H_
#define AVG_POOL_V2_GRAD_NHWC_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "avg_pool_v2_grad_base.h"
#include "avg_pool_v2_grad_tiling_data.h"

namespace AvgPoolV2GradNHWCNameSpace
{
using namespace AscendC;
using namespace AvgPoolV2Grad;
using computeType = float;

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
class AvgPoolV2GradKernelNHWC
{
public:
    __aicore__ inline AvgPoolV2GradKernelNHWC(TPipe* pipe, const AvgPoolV2GradNHWCTilingData* __restrict tilingData)
        : pipe_(pipe),tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCProcVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCMergeWProcVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,__local_mem__ uint32_t* helpAddr,
        __local_mem__ T3* helpAddrT3);
    __aicore__ inline void ProcessNoGradBlock();
    __aicore__ inline void CopyOut();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;
    TBuf<QuePosition::VECCALC> helpBufT3_;

    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> gradGm_;
    const AvgPoolV2GradNHWCTilingData* tilingData_;

    uint32_t blockIdx_ = 0;

    int64_t nOutputActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t cOutputActual_ = 1;
    int64_t cOutputAligned_ = 1;

    int64_t nAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;

    int64_t hGradActual_ = 0;
    int64_t wGradActual_ = 0;

    int64_t nOutputGradOffset_ = 0;
    int64_t hAxisGradOffset_ = 0;
    int64_t wAxisGradOffset_ = 0;
    int64_t cAxisGradOffset_ = 0;
    int64_t hGradActualStart_ = 0;
    int64_t wGradActualStart_ = 0;

    int64_t gradPlaneSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;
    int64_t curCoreProcessNum_ = 1;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T1 = platform::GetVRegSize() / sizeof(T1);
};

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Init(
    GM_ADDR grad, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    gradPlaneSize_ = tilingData_->hGrad * tilingData_->wGrad;

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_->usedCoreNum) ? tilingData_->tailCoreProcessNum : tilingData_->normalCoreProcessNum;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->outputBufferSize);
    pipe_->InitBuffer(gradQue_, BUFFER_NUM, tilingData_->inputGradBufferSize);
    pipe_->InitBuffer(helpBuf_, HELP_BUFFER);
    pipe_->InitBuffer(helpBufT3_, HELP_BUFFER_T3);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_->normalCoreProcessNum + loopNum;
    nAxisIndex_ = baseBlockIdx / (tilingData_->hOutputOuter * tilingData_->wOutputOuter * tilingData_->cOutputOuter);
    nOutputActual_ = nAxisIndex_ == (tilingData_->nOutputOuter - 1) ? tilingData_->nOutputTail : tilingData_->nOutputInner;

    int64_t tempNTail = baseBlockIdx % (tilingData_->hOutputOuter * tilingData_->wOutputOuter * tilingData_->cOutputOuter);
    cAxisIndex_ = tempNTail / (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    cOutputActual_ = cAxisIndex_ == (tilingData_->cOutputOuter - 1) ? tilingData_->cOutputTail : tilingData_->cOutputInner;
    cOutputAligned_ =
        (cOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t tempCTail = tempNTail % (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    hAxisIndex_ = tempCTail / (tilingData_->wOutputOuter);
    hOutputActual_ = hAxisIndex_ == (tilingData_->hOutputOuter - 1) ? tilingData_->hOutputTail : tilingData_->hOutputInner;

    wAxisIndex_ = tempCTail % tilingData_->wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_->wOutputOuter - 1) ? tilingData_->wOutputTail : tilingData_->wOutputInner;

    hGradActualStart_ = PStart(hAxisIndex_ * tilingData_->hOutputInner, tilingData_->padTop, tilingData_->hKernel, tilingData_->hStride);
    int64_t hGradActualEnd = PEnd(hAxisIndex_ * tilingData_->hOutputInner + hOutputActual_ - 1, tilingData_->padTop, tilingData_->hStride, tilingData_->hGrad);
    wGradActualStart_ = PStart(wAxisIndex_ * tilingData_->wOutputInner, tilingData_->padLeft, tilingData_->wKernel, tilingData_->wStride);
    int64_t wGradActualEnd = PEnd(wAxisIndex_ * tilingData_->wOutputInner + wOutputActual_ - 1, tilingData_->padLeft, tilingData_->wStride, tilingData_->wGrad);
    hGradActual_ = hGradActualEnd - hGradActualStart_;
    wGradActual_ = wGradActualEnd - wGradActualStart_;

    curHProBatchSize_ = tilingData_->hProBatchSize > hGradActual_ ? hGradActual_ : tilingData_->hProBatchSize;
    curWProBatchSize_ = tilingData_->wProBatchSize > wGradActual_ ? wGradActual_ : tilingData_->wProBatchSize;

    nOutputGradOffset_ = nAxisIndex_ * tilingData_->nOutputInner * gradPlaneSize_ * tilingData_->cOutput;
    hAxisGradOffset_ = hGradActualStart_ * tilingData_->wGrad * tilingData_->cOutput;
    wAxisGradOffset_ = wGradActualStart_ * tilingData_->cOutput;
    cAxisGradOffset_ = cAxisIndex_ * tilingData_->cOutputInner;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessPerLoop()
{
    if (hGradActual_ <= 0 || wGradActual_ <= 0) {
        ProcessNoGradBlock();  // ceilMode为false时，最后的尾块可能是这种情况
        return;
    }

    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();

    int64_t gradGmOffset = nOutputGradOffset_ + hAxisGradOffset_ + wAxisGradOffset_ + cAxisGradOffset_;

    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hGradActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = tilingData_->wGrad * tilingData_->cOutput * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = gradPlaneSize_ * tilingData_->cOutput * sizeof(T1);
    loopModeParamsT1.loop1DstStride = wGradActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = hGradActual_ * wGradActual_ * cOutputAligned_ * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(wGradActual_),
        static_cast<uint32_t>(cOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>((tilingData_->cOutput - cOutputActual_) * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(gradLocal, gradGm_[gradGmOffset], copyOutParamT1, paramsT1);

    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Compute()
{
    uint32_t calCount = tilingData_->outputBufferSize / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
    LocalTensor<T3> helpTensorT3 = helpBufT3_.Get<T3>();
    __local_mem__ T3* helpAddrT3 = (__local_mem__ T3*)helpTensorT3.GetPhyAddr();

    uint16_t computeSize = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSize / cOutputActual_;
    if (concurrencyCount < 2) {
        ConCProcVF<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr);
    } else {
        if constexpr (std::is_same<T3, int64_t>::value) {
            ConCMergeWProcVF<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
        } else {
            ConCMergeWProcVF<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
        }
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();

    int64_t outputPlaneSize = tilingData_->hOutput * tilingData_->wOutput * tilingData_->cOutput;
    int64_t nOutputAxisOffset = nAxisIndex_ * tilingData_->nOutputInner * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_->hOutputInner * tilingData_->wOutput * tilingData_->cOutput;
    int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_->wOutputInner * tilingData_->cOutput;
    int64_t cOutputAxisOffset = cAxisIndex_ * tilingData_->cOutputInner;
    int64_t outputGmOffset = nOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset + cOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hOutputActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = hOutputActual_ * wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = tilingData_->wOutput * tilingData_->cOutput * sizeof(T1);
    loopModeParamsT1.loop2DstStride = outputPlaneSize * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(wOutputActual_),
                                        static_cast<uint32_t>(cOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_->cOutput - cOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1>
__aicore__ inline void GetContinuousInput(MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          uint32_t gradOffset)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(gradRegT1, gradAddr + gradOffset);
        AscendC::MicroAPI::UnPack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1,
                                  (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopy(gradReg, gradAddr + gradOffset);
    }
}

template <typename T, const MicroAPI::RegTrait& Trait, const uint32_t COUNT_PAD>
__aicore__ inline void ComputeDivisor(
    MicroAPI::RegTensor<int32_t>& divisorW, MicroAPI::RegTensor<T, Trait>& outWStart,
    MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t wOutput, uint16_t padW, uint16_t padRightW, uint16_t kW,
    uint32_t count)
{
    uint32_t numT = count;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(numT);
    AscendC::MicroAPI::RegTensor<T, Trait> wStartReg;
    AscendC::MicroAPI::RegTensor<T, Trait> wEndReg;

    AscendC::MicroAPI::Adds(wStartReg, outWStart, T(-padW), maskT);
    AscendC::MicroAPI::Adds(wEndReg, wStartReg, T(kW), maskT);
    AscendC::MicroAPI::Mins(wEndReg, wEndReg, T(wOutput + padRightW), maskT);

    if constexpr (COUNT_PAD == 0) {
        AscendC::MicroAPI::Max(wStartReg, wStartReg, zeroConstRegT, maskT);
        AscendC::MicroAPI::Mins(wEndReg, wEndReg, wOutput, maskT);
    }

    AscendC::MicroAPI::Sub(
        divisorW, (AscendC::MicroAPI::RegTensor<int32_t>&)wEndReg, (AscendC::MicroAPI::RegTensor<int32_t>&)wStartReg,
        maskT);
}

template <typename T, const MicroAPI::RegTrait& Trait, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE,
    const uint32_t COUNT_PAD>
__aicore__ inline void GenDivisor(
    MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::RegTensor<T, Trait>& outWStart,
    MicroAPI::RegTensor<T, Trait>& outHStart, MicroAPI::RegTensor<T, Trait>& zeroConstRegT, int32_t hOutput,
    int32_t wOutput, uint16_t padH, uint16_t padW, uint16_t padDownH, uint16_t padRightW, uint16_t kH, uint16_t kW,
    int32_t divisorOverride, uint32_t count)
{
    if constexpr (HAS_DIVISOR == 1) {
        AscendC::MicroAPI::Duplicate(divisorReg, divisorOverride);
    } else if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> divisorW;
        AscendC::MicroAPI::RegTensor<int32_t> divisorH;
        uint32_t numI32 = count;
        AscendC::MicroAPI::MaskReg maskI32 = AscendC::MicroAPI::UpdateMask<int32_t>(numI32);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorW, outWStart, zeroConstRegT, wOutput, padW, padRightW, kW, count);
        ComputeDivisor<T, Trait, COUNT_PAD>(divisorH, outHStart, zeroConstRegT, hOutput, padH, padDownH, kH, count);
        AscendC::MicroAPI::Mul(divisorReg, divisorW, divisorH, maskI32);
    } else {
        AscendC::MicroAPI::Duplicate(divisorReg, int32_t(kH * kW));
    }
}

template <typename T>
__aicore__ inline void GradientAccBigC(
    __local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg,
    T scatterIndex, MicroAPI::RegTensor<int32_t> divisorReg, MicroAPI::MaskReg& pregRes)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorCastReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorResReg;
    AscendC::MicroAPI::DataCopy(scatterAccResReg, yAddr + scatterIndex);
    AscendC::MicroAPI::Cast<computeType, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, pregRes);
    AscendC::MicroAPI::Div(divisorResReg, gradReg, divisorCastReg, pregRes);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, divisorResReg, pregRes);

    AscendC::MicroAPI::DataCopy(yAddr + scatterIndex, scatterAccResReg, pregRes);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void DoSingleCNhwc(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                        uint32_t gradOffset, uint32_t gradMaskCount,
                                        int32_t hOutputActual, int32_t wOutputActual,
                                        int32_t cOutputAligned, int32_t cOffset, int32_t nOffset,
                                        int32_t hkStart, int32_t hkEnd, int32_t wkStart, int32_t wkEnd,
                                        MicroAPI::RegTensor<int32_t>& divisorReg, int32_t hIndex, int32_t wIndex)
{
    int32_t scatterIndex = 0;
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    uint32_t gradMask = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);
    GetContinuousInput(gradReg, gradAddr, gradOffset);
    int32_t scatterStartIndex = nOffset + hIndex * wOutputActual * cOutputAligned + wIndex * cOutputAligned + cOffset;

    for (uint16_t hIdx = hkStart; hIdx < hkEnd; hIdx++) {
        int32_t hKernelOffset = hIdx * wOutputActual * cOutputAligned;
        for (uint16_t wIdx = wkStart; wIdx < wkEnd; wIdx++) {
            int32_t scatterIndexOffsetTotal = hKernelOffset + wIdx * cOutputAligned;
            scatterIndex = scatterIndexOffsetTotal + scatterStartIndex;
            GradientAccBigC(yAddr, gradReg, scatterIndex, divisorReg, pregRes);
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulCNhwc(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
    uint32_t gradMaskCount, int32_t nOffset, int32_t wOutputActual, int32_t cOutputAligned,
    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
    MicroAPI::RegTensor<int32_t>& hMaxReg, uint16_t kH, uint16_t kW, MicroAPI::RegTensor<int32_t>& divisorReg,
    MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<int32_t>& hIndexReg,
    AscendC::MicroAPI::RegTensor<int32_t>& tmplWRegIdx)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterIndexReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregI32 = AscendC::MicroAPI::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    AscendC::MicroAPI::Muls(scatterStartIdxReg, hIndexReg, wOutputActual * cOutputAligned, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, wIndexReg, pregI32);

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        int32_t hKernelOffset = hIdx * wOutputActual * cOutputAligned;
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            uint32_t gradMask = gradMaskCount;
            AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);

            int32_t scatterIndexOffsetTotal = nOffset + hKernelOffset + wIdx * cOutputAligned;
            AscendC::MicroAPI::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);

            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::MicroAPI::RegTensor<int32_t> wCurIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hCurIndexReg;
                AscendC::MicroAPI::Adds(wCurIndexReg, tmplWRegIdx, int32_t(wIdx), pregRes);
                AscendC::MicroAPI::Adds(hCurIndexReg, hIndexReg, int32_t(hIdx), pregRes);
                FilterMask(pregRes, hCurIndexReg, wCurIndexReg, zeroConstReg, wMaxReg, hMaxReg);
            }

            GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void GenRepeatIndices(MicroAPI::RegTensor<T, Trait>& indexReg, uint16_t repeatCount, uint16_t repeatSize)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(repeatSize));
    AscendC::MicroAPI::Div(indexReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(repeatCount), preg);
}

template <typename T, const AscendC::MicroAPI::RegTrait& Trait = AscendC::MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeStridedIndices(
    MicroAPI::RegTensor<T, Trait>& outReg, MicroAPI::RegTensor<T, Trait>& inputGradRepeatReg, uint16_t cOutputActual,
    uint16_t cOutputAligned, int64_t curIndex, uint16_t padW)
{
    AscendC::MicroAPI::RegTensor<T, Trait> tmpReg;
    AscendC::MicroAPI::RegTensor<T, Trait> modReg;
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

    AscendC::MicroAPI::Arange(tmpReg, 0);
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(modReg, tmpReg, constReg, preg);
    AscendC::MicroAPI::Muls(modReg, modReg, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(tmpReg, tmpReg, modReg, preg);

    AscendC::MicroAPI::Muls(outReg, inputGradRepeatReg, static_cast<T>(cOutputAligned), preg);
    AscendC::MicroAPI::Add(outReg, outReg, tmpReg, preg);

    AscendC::MicroAPI::Adds(inputGradRepeatReg, inputGradRepeatReg, static_cast<T>(-curIndex - padW), preg);
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutWHIndex(
    MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<int32_t>& hIndexReg,
    MicroAPI::RegTensor<T, Trait>& outWStart, MicroAPI::RegTensor<T, Trait>& outHStart, int64_t curWIndex, int64_t curHIndex,
    uint16_t cOutputAligned, uint16_t padH, uint16_t padW, uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T, Trait> wIndexRegT;
    AscendC::MicroAPI::RegTensor<T, Trait> hIndexRegT;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(count);
    AscendC::MicroAPI::Adds(
        wIndexRegT, outWStart, static_cast<T>(-(curWIndex + padW) * cOutputAligned), maskT);
    AscendC::MicroAPI::Adds(hIndexRegT, outHStart, static_cast<T>(-curHIndex - padH), maskT);
    wIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)wIndexRegT;
    hIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)hIndexRegT;
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(
    MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate, int64_t colNum, int64_t fullBatchColNum,
    int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum * cOutputActual));

    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowGenRate * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(colGenRate * cOutputAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(
    MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNum, int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(cOutputActual));

    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNum * cOutputAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessNoGradBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(tilingData_->outputBufferSize) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr)
{
    int64_t wOutput = tilingData_->wOutput;
    int64_t hOutput = tilingData_->hOutput;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;

    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;

    uint16_t hGradActual = hGradActual_;
    uint16_t wGradActual = wGradActual_;
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t cRepeatimes = cOutputActual / computeSizeFP32;
    uint16_t cRemain = cOutputActual - cRepeatimes * computeSizeFP32;
    uint16_t cRemainLoopTimes = cRemain == 0 ? 0 : 1;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
        for (uint16_t hIdx = 0; hIdx < hGradActual; ++hIdx) {
            T3 hGradOffset = hIdx + hGradActualStart;
            for (uint16_t wIdx = 0; wIdx < wGradActual; ++wIdx) {
                T3 wGradOffset = wIdx + wGradActualStart;
                T3 hIndex = hGradOffset * strideH - curHIndex - padH;
                T3 wIndex = wGradOffset * strideW - curWIndex - padW;
                int32_t hkStart = hIndex > 0 ? 0 : (-hIndex);
                int32_t hkEnd = (hOutputActual - hIndex) > kH ? kH : (hOutputActual - hIndex);
                int32_t wkStart = wIndex > 0 ? 0 : (-wIndex);
                int32_t wkEnd = (wOutputActual - wIndex) > kW ? kW : (wOutputActual - wIndex);

                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::Duplicate(outHStart, T3(hGradOffset * strideH));
                    AscendC::MicroAPI::Duplicate(outWStart, T3(wGradOffset * strideW));
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));


                    for (uint16_t cIdx = 0; cIdx < cRepeatimes; ++cIdx) {
                        uint32_t cOffset = cIdx * computeSizeFP32;
                        uint32_t gradOffset = ((nIdx * hGradActual + hIdx) * wGradActual + wIdx) * cOutputAligned + cOffset;  // 当前grad的一维索引

                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, uint32_t(computeSizeFP32)
                        );

                        DoSingleCNhwc<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(yAddr, gradAddr, gradOffset, computeSizeFP32,
                                                                    hOutputActual, wOutputActual, cOutputAligned,
                                                                    cOffset, nOffset, hkStart, hkEnd, wkStart, wkEnd,
                                                                    divisorReg, hIndex, wIndex);
                    }

                    // cRemain
                    for (uint16_t cIdx = 0; cIdx < cRemainLoopTimes; ++cIdx) {
                        uint32_t cOffset = cRepeatimes * computeSizeFP32;
                        uint32_t gradOffset =
                            ((nIdx * hGradActual + hIdx) * wGradActual + wIdx) * cOutputAligned + cOffset;

                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, uint32_t(cRemain)
                        );

                        DoSingleCNhwc<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(yAddr, gradAddr, gradOffset, cRemain,
                                                                    hOutputActual, wOutputActual, cOutputAligned,
                                                                    cOffset, nOffset, hkStart, hkEnd, wkStart, wkEnd,
                                                                    divisorReg, hIndex, wIndex);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPoolV2GradKernelNHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCMergeWProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;

    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = static_cast<uint16_t>(hGradActual_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;

    uint16_t computeSizeFp32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFp32 / cOutputActual;

    uint16_t repeatimes = wFullBatchCount / concurrencyCount;
    uint16_t wRemain = wGradActual - repeatimes * wProBatchSize * concurrencyCount;
    uint32_t wRemainBatch = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;

    uint32_t mask0 = concurrencyCount * cOutputActual;
    uint32_t mask1 = wRemainBatch * cOutputActual;
    uint32_t mask2 = 1 * cOutputActual;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
        uint32_t nGradOffset = nIdx * hGradActual * wGradActual * cOutputAligned;
        for (uint16_t hIdx = 0; hIdx < hGradActual; hIdx++) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                AscendC::MicroAPI::MaskReg allMask =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg allMaskT3 =
                    AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

                GenInitial3DIndices<int32_t>(
                    (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, 1, wGradActual,
                    wFullBatchCount, cOutputActual, cOutputAligned);
                GenRepeatIndices<T3, Trait>(initialWRegIdx, wProBatchSize, cOutputActual);
                AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
                AscendC::MicroAPI::DataCopy(helpAddrT3, initialWRegIdx, allMaskT3);
            }

            for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        }

                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;

                        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        if constexpr (COUNT_PAD == 0) {
                            AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                        }

                        AscendC::MicroAPI::MaskReg allMask =
                            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::MaskReg allMaskT3 =
                            AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

                        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                        AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);

                        T3 hGradOffset = hIdx + hGradActualStart;
                        T3 wGradOffset = wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + wGradActualStart;
                        uint32_t offset = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + hIdx * wGradActual) *
                                          cOutputAligned + nGradOffset;

                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                        AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                        AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);

                        AscendC::MicroAPI::Duplicate(outHStart, T3(hGradOffset * strideH));

                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, tmplWRegIdx, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, mask0);
                        ComputeStridedIndices<T3, Trait>(
                            outWStart, tmplWRegIdx, cOutputActual, cOutputAligned, curWIndex, padW);
                        ComputeOutWHIndex<T3, Trait>(
                            wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, cOutputAligned, padH,
                            padW, mask0);

                        DoMulCNhwc<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask0, nOffset, wOutputActual, cOutputAligned,
                            zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                    }
                }
            }

            // 尾段整batch  用不满mask
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    }

                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;

                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                    }

                    AscendC::MicroAPI::MaskReg allMask =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg allMaskT3 =
                        AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

                    AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                    AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);

                    T3 hGradOffset = hIdx + hGradActualStart;
                    T3 wGradOffset = wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + wGradActualStart;
                    uint32_t offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + hIdx * wGradActual) *
                                    cOutputAligned + nGradOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                    AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                    AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);

                    AscendC::MicroAPI::Duplicate(outHStart, T3(hGradOffset * strideH));

                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, tmplWRegIdx, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask1);
                    ComputeStridedIndices<T3, Trait>(
                        outWStart, tmplWRegIdx, cOutputActual, cOutputAligned, curWIndex, padW);
                    ComputeOutWHIndex<T3, Trait>(
                        wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, cOutputAligned, padH,
                        padW, mask1);

                    DoMulCNhwc<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask1, nOffset, wOutputActual, cOutputAligned, zeroConstReg,
                        wMaxReg, hMaxReg, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                }
            }

            // 尾段零散点
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                }

                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;

                AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                }

                AscendC::MicroAPI::MaskReg allMask =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg allMaskT3 =
                    AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();

                AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);

                T3 hGradOffset = hIdx + hGradActualStart;
                AscendC::MicroAPI::Duplicate(outHStart, T3(hGradOffset * strideH));

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    uint32_t wGradOffset = wBatchIdx + wRemainBatch * wProBatchSize +
                                        repeatimes * concurrencyCount * wProBatchSize + wGradActualStart;
                    uint32_t offset = (wBatchIdx + wRemainBatch * wProBatchSize +
                                    repeatimes * concurrencyCount * wProBatchSize + hIdx * wGradActual) *
                                    cOutputAligned + nGradOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                    AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                    AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, tmplWRegIdx, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask2);
                    ComputeStridedIndices<T3, Trait>(
                        outWStart, tmplWRegIdx, cOutputActual, cOutputAligned, curWIndex, padW);
                    ComputeOutWHIndex<T3, Trait>(
                        wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, cOutputAligned, padH,
                        padW, mask2);

                    DoMulCNhwc<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask2, nOffset, wOutputActual, cOutputAligned, zeroConstReg,
                        wMaxReg, hMaxReg, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                }
            }
        }
    }
}
}  // namespace AvgPoolV2GradNHWCNameSpace
#endif  // AVG_POOL_V2_GRAD_NHWC_KERNEL_H_
