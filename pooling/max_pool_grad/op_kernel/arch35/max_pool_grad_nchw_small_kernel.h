/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_GRAD_SMALL_KERNEL_H
#define MAX_POOL_GRAD_SMALL_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_grad_struct.h"
#include "max_pool_grad_nchw_backward_base.h"
#include "../../max_pool_with_argmax_v3/arch35/max_pool_with_argmax_v3_base.h"
#include "../pool_3d_common/arch35/pool_big_kernel_utils.h"

namespace MaxPoolGradNCHWSmallKernelNameSpace {
using namespace AscendC;
using MaxPoolGradNCHWNameSpace::GenInitial1DIndices;
using MaxPoolGradNCHWNameSpace::GenInitial2DIndices;
using MaxPoolGradNCHWNameSpace::GenInitial3DIndices;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t RATIO = 2;

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
class PoolGradNCHWSmallKernel : public MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<
                                    TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE> {
    using Base =
        MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>;

public:
    __aicore__ inline PoolGradNCHWSmallKernel(
        TPipe& pipeIn, const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ForwardScalarCompute();
    __aicore__ inline void ConvertIndexWithoutPadAlign(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
        TYPE_ARGMAX hIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset);
    __aicore__ inline void ProcessW(
        __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset,
        MicroAPI::RegTensor<int32_t>& indexReg, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
        MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation, uint32_t wDilation);
    __aicore__ inline void ConvertIndexWithoutPadAlignNc(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
        TYPE_ARGMAX hIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount,
        int32_t inputNcSize);
    __aicore__ inline void MultiRowGather(
        __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void SingleRowGather(
        __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void MultiNcGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void DupBufferNegInf(
        __local_mem__ TYPE_ORIG_X* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail);
    __aicore__ inline void CopyToCalcBuffer(
        __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr, uint16_t batch, uint16_t rows,
        uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcRowStride,
        uint32_t dstBatchStride, uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset);
    __aicore__ inline void DupAndCopyToCalcBuffer(
        __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr);
    __aicore__ inline void ForwardCopyIn();
    __aicore__ inline void Forward();
    __aicore__ inline void Backward();

    TPipe& pipe_;
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;
    TBuf<TPosition::VECCALC> argmaxBuff_;

    GlobalTensor<TYPE_ORIG_X> xGm_;

    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;
    int64_t wInputActualAlignedPad_ = 0;
    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;
    int64_t highInputOffset_ = 0;
    int64_t forwardhInputOffset_ = 0;
    int64_t forwardwInputOffset_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
    int64_t forwardHighAxisIndex_ = 0;
    int64_t forwardhighAxisActual_ = 0;
    int64_t forwardHAxisIndex_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t wOutputReal_ = 0;
    bool isPad_ = false;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(TYPE_ORIG_X) >= BLOCK_SIZE / sizeof(TYPE_ARGMAX) ? BLOCK_SIZE / sizeof(TYPE_ORIG_X) :
                                                                               BLOCK_SIZE / sizeof(TYPE_ARGMAX);
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(TYPE_ORIG_X);
};

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Init(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y)
{
    Base::ParseTilingData(tilingData_);
    Base::blockIdx_ = GetBlockIdx();
    if (Base::blockIdx_ >= Base::usedCoreNum_) {
        return;
    }
    Base::curCoreProcessNum_ =
        (Base::blockIdx_ + 1 == Base::usedCoreNum_) ? Base::tailCoreProcessNum_ : Base::normalCoreProcessNum_;
    xGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)orig_x);
    Base::gradGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)grads);
    Base::yGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)y);

    isPad_ = tilingData_.isPad != 0;

    pipe_.InitBuffer(inputQue_, BUFFER_NUM, tilingData_.inputBufferSize);
    if (isPad_) {
        pipe_.InitBuffer(inputCalcBuff_, tilingData_.inputBufferSize);
    }

    pipe_.InitBuffer(argmaxBuff_, Base::argmaxBufferSize_);
    pipe_.InitBuffer(Base::outputQue_, BUFFER_NUM, Base::outputBufferSize_);
    pipe_.InitBuffer(Base::gradQue_, BUFFER_NUM, Base::gradBufferSize_);
    pipe_.InitBuffer(Base::helpBuf_, HELP_BUFFER);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Forward()
{
    LocalTensor<TYPE_ORIG_X> inputLocal = inputQue_.template DeQue<TYPE_ORIG_X>();
    __local_mem__ TYPE_ORIG_X* inputQueAddr = (__local_mem__ TYPE_ORIG_X*)inputLocal.GetPhyAddr();
    __local_mem__ TYPE_ORIG_X* computeAddr = inputQueAddr;
    if (isPad_) {
        LocalTensor<TYPE_ORIG_X> caclBuffLocal = inputCalcBuff_.template Get<TYPE_ORIG_X>();
        __local_mem__ TYPE_ORIG_X* inputBuffAddr = (__local_mem__ TYPE_ORIG_X*)caclBuffLocal.GetPhyAddr();
        DupAndCopyToCalcBuffer(inputBuffAddr, inputQueAddr);
        computeAddr = inputBuffAddr;
    }
    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.template Get<TYPE_ARGMAX>();
    __local_mem__ TYPE_ARGMAX* argmaxAddr = (__local_mem__ TYPE_ARGMAX*)argmaxLocal.GetPhyAddr();

    if (Base::wArgmaxActual_ * RATIO > Base::vlT2_) {
        SingleRowGather(computeAddr, argmaxAddr);
    } else if (Base::hArgmaxActual_ * Base::wArgmaxActual_ * RATIO > Base::vlT2_) {
        MultiRowGather(computeAddr, argmaxAddr);
    } else {
        MultiNcGather(computeAddr, argmaxAddr);
    }

    inputQue_.FreeTensor(inputLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Backward()
{
    uint32_t calCount = Base::outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = Base::outputQue_.template AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    LocalTensor<TYPE_ORIG_X> gradLocal = Base::gradQue_.template DeQue<TYPE_ORIG_X>();
    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.template Get<TYPE_ARGMAX>();
    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ TYPE_ORIG_X* gradAddr = (__local_mem__ TYPE_ORIG_X*)gradLocal.GetPhyAddr();
    __local_mem__ TYPE_ARGMAX* argmaxAddr = (__local_mem__ TYPE_ARGMAX*)argmaxLocal.GetPhyAddr();

    Base::BackwardCompute(yAddr, gradAddr, argmaxAddr);

    if constexpr (std::negation<std::is_same<TYPE_ORIG_X, float>>::value) {
        Cast(yLocal.ReinterpretCast<TYPE_ORIG_X>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    Base::outputQue_.EnQue(yLocal);
    Base::gradQue_.FreeTensor(gradLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ForwardScalarCompute()
{
    forwardHighAxisIndex_ = Base::highAxisIndex_;
    forwardhighAxisActual_ = Base::highAxisActual_;

    forwardHAxisIndex_ = Base::hAxisIndex_;
    hOutputReal_ = Base::hArgmaxActual_;

    wOutputReal_ = Base::wArgmaxActual_;

    hInputActualPad_ = (hOutputReal_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.dilationH + 1;
    wInputActualPad_ = (wOutputReal_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.dilationW + 1;

    wInputActualAlignedPad_ =
        CeilDivision(wInputActualPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) * (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
    int64_t inputPlaneSize = tilingData_.hOutput * tilingData_.wOutput;
    highInputOffset_ = Base::highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    forwardhInputOffset_ = Base::hArgmaxActualStart_ * tilingData_.hStride * tilingData_.wOutput;
    forwardwInputOffset_ = Base::wArgmaxActualStart_ * tilingData_.wStride;

    if (isPad_) {
        int64_t tRelBoundDistance = Base::hArgmaxActualStart_ * tilingData_.hStride - tilingData_.padH;
        int64_t bRelBoundDistance = Base::hArgmaxActualStart_ * tilingData_.hStride +
                                    (hOutputReal_ - 1) * tilingData_.hStride + tilingData_.hKernel -
                                    tilingData_.hOutput - tilingData_.padH;
        int64_t lRelBoundDistance = Base::wArgmaxActualStart_ * tilingData_.wStride - tilingData_.padW;
        int64_t rRelBoundDistance = Base::wArgmaxActualStart_ * tilingData_.wStride +
                                    (wOutputReal_ - 1) * tilingData_.wStride + tilingData_.wKernel -
                                    tilingData_.wOutput - tilingData_.padW;
        leftOffsetToInputLeft_ = lRelBoundDistance >= 0 ? 0 : -lRelBoundDistance;
        rightOffsetToInputRight_ = rRelBoundDistance >= 0 ? rRelBoundDistance : 0;
        topOffsetToInputTop_ = tRelBoundDistance >= 0 ? 0 : -tRelBoundDistance;
        downOffsetToInputDown_ = bRelBoundDistance >= 0 ? bRelBoundDistance : 0;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
        forwardhInputOffset_ =
            topOffsetToInputTop_ == 0 ? forwardhInputOffset_ - tilingData_.padH * tilingData_.wOutput : 0;
        forwardwInputOffset_ = leftOffsetToInputLeft_ == 0 ? forwardwInputOffset_ - tilingData_.padW : 0;
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ForwardCopyIn()
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.template AllocTensor<TYPE_ORIG_X>();
    int64_t xGmOffset = highInputOffset_ + forwardhInputOffset_ + forwardwInputOffset_;

    LoopModeParams loopModeParams;
    if (isPad_) {
        int64_t wInputActualAlignedNoPad =
            CeilDivision(wInputActualNoPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) * (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
        loopModeParams.loop1Size = Base::highAxisActual_;
        loopModeParams.loop1SrcStride = Base::hOutput_ * Base::wOutput_ * sizeof(TYPE_ORIG_X);
        loopModeParams.loop1DstStride = hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(TYPE_ORIG_X);
        loopModeParams.loop2Size = 1;
        loopModeParams.loop2SrcStride = 0;
        loopModeParams.loop2DstStride = 0;
    } else {
        loopModeParams.loop1Size = Base::highAxisActual_;
        loopModeParams.loop1SrcStride = Base::hOutput_ * Base::wOutput_ * sizeof(TYPE_ORIG_X);
        loopModeParams.loop1DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(TYPE_ORIG_X);
        loopModeParams.loop2Size = 1;
        loopModeParams.loop2SrcStride = 0;
        loopModeParams.loop2DstStride = 0;
    }
    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<TYPE_ORIG_X> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams;
    if (isPad_) {
        copyParams.blockCount = static_cast<uint16_t>(hInputActualNoPad_);
        copyParams.blockLen = static_cast<uint32_t>(wInputActualNoPad_ * sizeof(TYPE_ORIG_X));
        copyParams.srcStride = static_cast<uint32_t>((Base::wOutput_ - wInputActualNoPad_) * sizeof(TYPE_ORIG_X));
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
    } else {
        copyParams.blockCount = static_cast<uint16_t>(hInputActualPad_);
        copyParams.blockLen = static_cast<uint32_t>(wInputActualPad_ * sizeof(TYPE_ORIG_X));
        copyParams.srcStride = static_cast<uint32_t>((Base::wOutput_ - wInputActualPad_) * sizeof(TYPE_ORIG_X));
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
    }
    DataCopyPad(xLocal, xGm_[xGmOffset], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Process()
{
    if (Base::blockIdx_ >= Base::usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; loopNum++) {
        Base::ScalarCompute(loopNum);
        PipeBarrier<PIPE_ALL>();
        if (Base::hArgmaxActual_ <= 0 || Base::wArgmaxActual_ <= 0) {
            Base::ProcessNoArgmaxBlock();
            continue;
        }
        ForwardScalarCompute();
        ForwardCopyIn();
        Forward();
        Base::CopyInGrad();
        Backward();
        Base::CopyOut();
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void
PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlign(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
    TYPE_ARGMAX hIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset)
{
    if (isPad_) {
        ConvertIndexWithoutPadAlignCommon<TYPE_ARGMAX, 1>(
            srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset);
    } else {
        ConvertIndexWithoutPadAlignCommon<TYPE_ARGMAX, 0>(
            srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset);
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ProcessW(
    __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset,
    MicroAPI::RegTensor<int32_t>& indexReg, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
    MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation, uint32_t wDilation)
{
    MicroAPI::RegTensor<int32_t> indexWithOffset;
    MicroAPI::RegTensor<TYPE_ORIG_X> calcReg;
    MicroAPI::RegTensor<int32_t> calcMaxIndexReg;
    uint32_t maskCount = repeatElem;
    MicroAPI::MaskReg allMaskU32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg gatherMask = MicroAPI::UpdateMask<TYPE_ORIG_X>(maskCount);
    MicroAPI::RegTensor<TYPE_ORIG_X> maxReg;
    MicroAPI::MaskReg neMask;
    MicroAPI::MaskReg gtMask;
    MicroAPI::MaskReg tmpMask;
    MicroAPI::UnalignReg u0;
    DuplicateNegInfReg<TYPE_ORIG_X>(maxReg);

    MicroAPI::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);
    for (int32_t hIndex = 0; hIndex < hKernel; hIndex++) {
        for (int32_t wIndex = 0; wIndex < wKernel; wIndex++) {
            int32_t relIndex = hIndex * wStrideOffset * hDilation + wIndex * wDilation;
            int32_t offset = static_cast<int32_t>(hOffset + relIndex);
            MicroAPI::Adds(indexWithOffset, indexReg, offset, allMaskU32);
            if constexpr (std::is_same<TYPE_ORIG_X, float>::value) {
                MicroAPI::DataCopyGather(
                    calcReg, computeAddr, (MicroAPI::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
            } else {
                MicroAPI::RegTensor<uint16_t> indexConvert;
                MicroAPI::Pack(indexConvert, indexWithOffset);
                MicroAPI::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
            }

            MicroAPI::Compare<TYPE_ORIG_X, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
            MicroAPI::Compare<TYPE_ORIG_X, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
            MicroAPI::MaskOr(gtMask, gtMask, neMask, gatherMask);

            if constexpr (sizeof(int32_t) / sizeof(TYPE_ORIG_X) == 1) {
                MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
            } else {
                MicroAPI::MaskUnPack(tmpMask, gtMask);
                MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, tmpMask);
            }
            MicroAPI::Max(maxReg, maxReg, calcReg, gatherMask);
        }
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void
PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlignNc(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
    TYPE_ARGMAX hIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount,
    int32_t inputNcSize)
{
    if (isPad_) {
        ConvertIndexWithoutPadAlignNcCommon<TYPE_ARGMAX, 1>(
            srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset, ncOutputCount, inputNcSize);
    } else {
        ConvertIndexWithoutPadAlignNcCommon<TYPE_ARGMAX, 0>(
            srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset, ncOutputCount, inputNcSize);
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::SingleRowGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t loopW = static_cast<uint16_t>(Base::wArgmaxActual_) / Base::vlT2_;
    uint16_t repeatsElem = Base::vlT2_;
    uint16_t tailRepeatsElem = static_cast<uint16_t>(Base::wArgmaxActual_) - loopW * Base::vlT2_;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }
    uint16_t hKernel = Base::kernelH_;
    uint16_t wKernel = Base::kernelW_;
    uint32_t wStride = Base::strideW_;
    TYPE_ARGMAX left = static_cast<TYPE_ARGMAX>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    TYPE_ARGMAX hIndexBase = static_cast<TYPE_ARGMAX>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    TYPE_ARGMAX wInput = static_cast<TYPE_ARGMAX>(Base::wOutput_);
    uint32_t highAxisActual = static_cast<uint32_t>(forwardhighAxisActual_);
    uint32_t hOutputActual = static_cast<uint32_t>(Base::hArgmaxActual_);
    uint32_t wOutputActual = static_cast<uint32_t>(Base::wArgmaxActual_);
    uint32_t hInputActualPad = static_cast<uint32_t>(hInputActualPad_);
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;

    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<int32_t> indexReg;
                MicroAPI::RegTensor<int32_t> maxIndexReg;
                MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
                MicroAPI::UnalignReg u1;
                MicroAPI::Arange(indexReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::MaskReg allMaskI32 =
                    AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                MicroAPI::Muls(indexReg, indexReg, static_cast<int32_t>(wStride), allMaskI32);
                int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
                int32_t ncOutOffset = nc * hOutputActual * wOutputActual;
                int32_t vfMaxAddrOffset = ncOutOffset + hLoop * wOutputActual;
                __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr + vfMaxAddrOffset;
                for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                    int32_t wOffset =
                        ncInputOffset + hLoop * wInputActualAlignedPad * Base::strideH_ + wLoop * repeatsElem * wStride;
                    ProcessW(
                        computeAddr, wOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                        repeatsElem, maxIndexReg, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(
                        maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                        maxIndexConvertReg, ncInputOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                }
                int32_t wOffsetTail =
                    ncInputOffset + hLoop * wInputActualAlignedPad * Base::strideH_ + loopW * repeatsElem * wStride;
                ProcessW(
                    computeAddr, wOffsetTail, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                    tailRepeatsElem, maxIndexReg, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(
                    maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                    maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::MultiRowGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint32_t wOutputActual = static_cast<uint32_t>(Base::wArgmaxActual_);
    uint16_t hKernel = Base::kernelH_;
    uint16_t wKernel = Base::kernelW_;
    uint32_t wStride = Base::strideW_;
    uint32_t rate2D = wInputActualAlignedPad_ * Base::strideH_;
    uint16_t hBatchCount = Base::vlT2_ / wOutputActual;
    uint16_t hLoopTimes = static_cast<uint16_t>(Base::hArgmaxActual_) / hBatchCount;
    uint16_t hTail = static_cast<uint16_t>(Base::hArgmaxActual_) - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputActual;
    uint16_t tailRepeatsElem = hTail * wOutputActual;
    TYPE_ARGMAX left = static_cast<TYPE_ARGMAX>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    TYPE_ARGMAX hIndexBase = static_cast<TYPE_ARGMAX>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    TYPE_ARGMAX wInput = static_cast<TYPE_ARGMAX>(Base::wOutput_);
    uint32_t highAxisActual = static_cast<uint32_t>(Base::highAxisActual_);
    uint32_t hInputActualPad = static_cast<uint32_t>(hInputActualPad_);
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hOutputActual = static_cast<uint32_t>(Base::hArgmaxActual_);
    uint32_t hStride = Base::strideH_;
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr;
        GenGatterIndex2D<int32_t>(
            indexReg, static_cast<int32_t>(rate2D), static_cast<int32_t>(wOutputActual), static_cast<int32_t>(wStride));
        for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
            int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
            for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                int32_t wOffset = ncInputOffset + hLoop * hBatchCount * hStride * wInputActualAlignedPad;
                ProcessW(
                    computeAddr, wOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                    repeatsElem, maxIndexReg, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(
                    maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                    maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
            int32_t wOffsetTail = ncInputOffset + hLoopTimes * hBatchCount * hStride * wInputActualAlignedPad;
            ProcessW(
                computeAddr, wOffsetTail, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                tailRepeatsElem, maxIndexReg, hDilation, wDilation);
            ConvertIndexWithoutPadAlign(
                maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                maxIndexConvertReg, ncInputOffset);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::MultiNcGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    uint16_t wKernel = Base::kernelW_;
    uint16_t hKernel = Base::kernelH_;
    uint32_t wStride = Base::strideW_;
    uint16_t rate3D = hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = static_cast<uint16_t>(Base::hArgmaxActual_ * Base::wArgmaxActual_);
    uint16_t rate2D = Base::strideH_ * wInputActualAlignedPad_;
    uint16_t wOutputActual = static_cast<uint16_t>(Base::wArgmaxActual_);
    uint16_t eachBatchCount = static_cast<uint16_t>(Base::hArgmaxActual_ * Base::wArgmaxActual_);
    uint16_t ncBatchCount = Base::vlT2_ / eachBatchCount;
    uint16_t ncLoopTimes = static_cast<uint16_t>(forwardhighAxisActual_) / ncBatchCount;
    uint16_t ncTail = static_cast<uint16_t>(forwardhighAxisActual_) - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;
    TYPE_ARGMAX left = static_cast<TYPE_ARGMAX>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    TYPE_ARGMAX hIndexBase = static_cast<TYPE_ARGMAX>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    TYPE_ARGMAX wInput = static_cast<TYPE_ARGMAX>(Base::wOutput_);
    uint32_t hInputActualPad = static_cast<uint32_t>(hInputActualPad_);
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hOutputActual = static_cast<uint32_t>(Base::hArgmaxActual_);
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<TYPE_ARGMAX> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ TYPE_ARGMAX* argmaxAddrLocal = argmaxAddr;
        GenGatterIndex3D<int32_t>(
            indexReg, static_cast<int32_t>(rate3D), static_cast<int32_t>(num2D), static_cast<int32_t>(rate2D),
            static_cast<int32_t>(wOutputActual), static_cast<int32_t>(wStride));
        for (uint16_t nc = 0; nc < ncLoopTimes; nc++) {
            uint32_t ncInputOffset = nc * ncBatchCount * hInputActualPad * wInputActualAlignedPad;
            int32_t hOffset = static_cast<int32_t>(nc) * ncBatchCount * rate3D;
            ProcessW(
                computeAddr, hOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                repeatsElem, maxIndexReg, hDilation, wDilation);
            ConvertIndexWithoutPadAlignNc(
                maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                maxIndexConvertReg, ncInputOffset, num2D, rate3D);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
        uint32_t ncInputOffsetTail = ncLoopTimes * ncBatchCount * hInputActualPad * wInputActualAlignedPad;
        int32_t hOffset = static_cast<int32_t>(ncLoopTimes) * ncBatchCount * rate3D;
        ProcessW(
            computeAddr, hOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
            tailRepeatsElem, maxIndexReg, hDilation, wDilation);
        ConvertIndexWithoutPadAlignNc(
            maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase, maxIndexConvertReg,
            ncInputOffsetTail, num2D, rate3D);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::DupBufferNegInf(
    __local_mem__ TYPE_ORIG_X* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    DupBufferNegInfCommon<TYPE_ORIG_X>(dstAddr, repeatElm, loop, tail);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyToCalcBuffer(
    __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr, uint16_t batch, uint16_t rows,
    uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcRowStride,
    uint32_t dstBatchStride, uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    CopyToCalcBuffer2DCommon<TYPE_ORIG_X>(
        dstAddr, srcAddr, batch, rows, loopCols, tailCols, repeatElm, srcBatchStride, srcRowStride, dstBatchStride,
        dstRowStride, dstRowOffset, dstColOffset);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::DupAndCopyToCalcBuffer(
    __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr)
{
    uint32_t wInputPadActualAlign =
        CeilDivision(wInputActualPad_, MAX_DATA_NUM_IN_ONE_BLOCK) * MAX_DATA_NUM_IN_ONE_BLOCK;
    uint16_t hPad = topOffsetToInputTop_;
    uint16_t wPad = leftOffsetToInputLeft_;
    uint16_t hRows = hInputActualNoPad_;
    uint16_t wCols = wInputActualNoPad_;
    uint16_t highAxis = Base::highAxisActual_;
    uint32_t srcBatchStride =
        hInputActualNoPad_ * CeilDivision(wInputActualNoPad_, MAX_DATA_NUM_IN_ONE_BLOCK) * MAX_DATA_NUM_IN_ONE_BLOCK;
    uint32_t dstBatchStride = (hRows + hPad + downOffsetToInputDown_) * wInputPadActualAlign;
    uint32_t srcRowStride = CeilDivision(wInputActualNoPad_, MAX_DATA_NUM_IN_ONE_BLOCK) * MAX_DATA_NUM_IN_ONE_BLOCK;
    uint32_t dstRowStride = wInputPadActualAlign;
    uint32_t repeatElm = MAX_DATA_NUM_IN_ONE_BLOCK;
    uint16_t loopCols = wCols / repeatElm;
    uint16_t tailCols = wCols % repeatElm;
    __VEC_SCOPE__
    {
        DupBufferNegInf(
            dstAddr, repeatElm, highAxis * (hRows + hPad + downOffsetToInputDown_) * (wInputPadActualAlign / repeatElm),
            repeatElm);
        CopyToCalcBuffer(
            dstAddr, srcAddr, highAxis, hRows, loopCols, tailCols, repeatElm, srcBatchStride, srcRowStride,
            dstBatchStride, dstRowStride, hPad, wPad);
    }
};

} // namespace MaxPoolGradNCHWSmallKernelNameSpace
#endif // MAX_POOL_GRAD_SMALL_KERNEL_H