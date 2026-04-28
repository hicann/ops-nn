/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_
#define MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_grad_struct.h"
#include "../inc/platform.h"
#include "../pool_grad_common/arch35/max_pool_grad_with_argmax_base_common.h"
#include "../pool_grad_common/arch35/max_pool_grad_nchw_scatter_common.h"

namespace MaxPoolGradNCHWBackwardBaseNameSpace {
using namespace AscendC;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;

using ::PEnd;
using ::PStart;
using MaxPoolGradNCHWNameSpace::DoMulNCNchw;
using MaxPoolGradNCHWNameSpace::DoSingleNCNchw;
using MaxPoolGradNCHWNameSpace::Gen2DIndexOne;
using MaxPoolGradNCHWNameSpace::Gen3DIndexOne;
using MaxPoolGradNCHWNameSpace::GenInitial1DIndices;
using MaxPoolGradNCHWNameSpace::GenInitial2DIndices;
using MaxPoolGradNCHWNameSpace::GenInitial3DIndices;

constexpr int32_t BK_BUFFER_NUM = 2;
constexpr int64_t RATIO = 2;
constexpr int32_t HELP_BUFFER = 4096;
constexpr uint32_t DOUBLE = 2;

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
class MaxPoolGradNCHWBackwardBase {
public:
    __aicore__ inline MaxPoolGradNCHWBackwardBase()
    {}

protected:
    __aicore__ inline void ParseTilingData(const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void BackwardCompute(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void singleLineProcessVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void multipleLineHwProcessVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void multipleLineProcessVF2(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);

    TQue<QuePosition::VECIN, BK_BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BK_BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> argmaxBuf_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;

    uint32_t blockIdx_ = 0;

    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;
    int64_t kernelH_ = 1;
    int64_t kernelW_ = 1;
    int64_t strideH_ = 1;
    int64_t strideW_ = 1;
    int64_t padH_ = 0;
    int64_t padW_ = 0;
    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;
    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;
    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;
    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;
    int64_t inputBufferSize_ = 1;
    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;
    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;
    int64_t wArgmaxAligned_ = 0;
    int64_t hArgmaxActualStart_ = 0;
    int64_t wArgmaxActualStart_ = 0;
    int64_t highAxisArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;
    int64_t argmaxPlaneSize_ = 1;
    int64_t hProBatchSize_ = 1;
    int64_t wProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1);
    constexpr static uint16_t vlT2_ = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
{
    hArgmax_ = tilingData.hArgmax;
    wArgmax_ = tilingData.wArgmax;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;
    kernelH_ = tilingData.hKernel;
    kernelW_ = tilingData.wKernel;
    strideH_ = tilingData.hStride;
    strideW_ = tilingData.wStride;
    padH_ = tilingData.padH;
    padW_ = tilingData.padW;
    dilationH_ = tilingData.dilationH;
    dilationW_ = tilingData.dilationW;
    highAxisInner_ = tilingData.highAxisInner;
    highAxisTail_ = tilingData.highAxisTail;
    highAxisOuter_ = tilingData.highAxisOuter;
    hOutputInner_ = tilingData.hOutputInner;
    hOutputTail_ = tilingData.hOutputTail;
    hOutputOuter_ = tilingData.hOutputOuter;
    wOutputInner_ = tilingData.wOutputInner;
    wOutputTail_ = tilingData.wOutputTail;
    wOutputOuter_ = tilingData.wOutputOuter;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    inputBufferSize_ = tilingData.inputBufferSize;
    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;
    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
    argmaxPlaneSize_ = hArgmax_ * wArgmax_;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    highAxisIndex_ = baseBlockIdx / (hOutputOuter_ * wOutputOuter_);
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;
    int64_t tempTail = baseBlockIdx % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempTail / wOutputOuter_;
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;
    wAxisIndex_ = tempTail % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;
    wOutputAligned_ =
        (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    int64_t hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    int64_t wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    int64_t wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;
    wArgmaxAligned_ =
        (wArgmaxActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;
    hArgmaxActualStart_ = hArgmaxActualStart;
    wArgmaxActualStart_ = wArgmaxActualStart;
    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;
    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::CopyInGrad()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = argmaxPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T1);
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop2DstStride = 0;
    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hArgmaxActual_), static_cast<uint32_t>(wArgmaxActual_ * sizeof(T1)),
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T1)), 0, 0};
    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t highOutputAxisOffset = highAxisIndex_ * highAxisInner_ * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wOutputAxisOffset = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = outputPlaneSize * sizeof(T1);
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop2DstStride = 0;
    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hOutputActual_), static_cast<uint32_t>(wOutputActual_ * sizeof(T1)), 0,
        static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T1)), 0};
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::BackwardCompute(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
    if (wConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        singleLineProcessVF(yAddr, gradAddr, argmaxAddr, helpAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        multipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr, helpAddr);
    } else {
        multipleLineProcessVF2(yAddr, gradAddr, argmaxAddr, helpAddr);
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::singleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxActual = wArgmaxActual_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t repeatimes = wFullBatchCount / computeSizeT2;
    uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * computeSizeT2;
    uint32_t wRemainBatchCount = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;
    uint32_t one = 1;
    uint32_t all = computeSizeT2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize);
        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highIdx * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset =
                            (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned +
                             highArgmaxOffset);
                        uint32_t indexOffset =
                            (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize + hIdx * wArgmaxActual +
                             highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, all, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    uint32_t offset =
                        (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned +
                         highArgmaxOffset);
                    uint32_t indexOffset =
                        (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxActual +
                         highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, wRemainBatchCount,
                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg,
                        hMaxReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    uint32_t offset =
                        (wBatchIdx + wRemainBatchCount * wProBatchSize + repeatimes * computeSizeT2 * wProBatchSize +
                         hIdx * wArgmaxAligned + highArgmaxOffset);
                    uint32_t indexOffset =
                        (wBatchIdx + wRemainBatchCount * wProBatchSize + repeatimes * computeSizeT2 * wProBatchSize +
                         hIdx * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, one, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::multipleLineHwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;

    uint16_t hConcurrentCount = V_REG_SIZE / (wFullBatchCount * sizeof(T2));

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t blockOne = 1 * hConcurrentCount;
    uint32_t remainBatchOne = 1 * hRemainBatchCount;
    uint32_t remainTailOne = 1;
    uint32_t maskBlock = wFullBatchCount * hConcurrentCount;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount;
    uint32_t maskRemainTail = wFullBatchCount;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        GenInitial2DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize, hProBatchSize, wArgmaxAligned,
            wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGradOne, hProBatchSize, wArgmaxAligned);
        GenInitial2DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
            wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highIdx * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                             hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        T2 indexOffset =
                            (wBatchIdx + hProBatchIdx * wArgmaxActual +
                             hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskBlock, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }

                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                             hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        T2 indexOffset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                             hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, blockOne, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + hProBatchIdx * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainBatch,
                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg,
                        hMaxReg);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainBatchOne, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                         hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + hProBatchIdx * wArgmaxActual + hRemainBatchCount * hProBatchSize * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainTail, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                         hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                         hRemainBatchCount * hProBatchSize * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainTailOne, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int32_t highOutputPlaneActual = wOutputAligned * hOutputActual;
    int64_t highAxisActual = highAxisActual_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t highConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(T2));

    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;

    uint16_t hRemainTail = hArgmaxActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = highConcurrentCount * whFullBatchCount;
    uint32_t mask1 = highConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = highConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * 1 * 1;
    uint32_t mask4 = highBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = highBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = highBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = highBlockRemainTail * 1 * 1;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        GenInitial3DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, wProBatchSize, hProBatchSize, wArgmaxAligned,
            wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, hProBatchSize, wArgmaxAligned, hFullBatchCount,
            hArgmaxActual);
        GenInitial2DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGrad, wProBatchSize, hArgmaxActual, wArgmaxAligned,
            wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGradOne, hArgmaxActual, wArgmaxAligned);
        GenInitial3DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
            wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wArgmaxActual, hFullBatchCount,
            hArgmaxActual);
        GenInitial2DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hArgmaxActual, wArgmaxActual,
            wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual, wArgmaxActual);
        AscendC::MicroAPI::MaskReg allMaskStore =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initial3DRegGrad, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegGradOne, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t), initial2DRegGrad, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t), initial2DRegGradOne, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndex, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMaskStore);
        AscendC::MicroAPI::DataCopy(helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMaskStore);
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(initial3DRegGrad, helpAddr);
        AscendC::MicroAPI::DataCopy(initial3DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask0, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, whFullBatchCount);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                         highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask1, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, hFullBatchCount);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(initial2DRegGrad, helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegGradOne, helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                         highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual +
                         highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask2, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, wFullBatchCount);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount +
                         (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                    T2 indexOffset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount +
                         (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual + highIndexOffset);
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask3, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, 1);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(initial3DRegGrad, helpAddr);
        AscendC::MicroAPI::DataCopy(initial3DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t));

        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                T2 indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + highIndexOffset);
                AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask4, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    whFullBatchCount);
            }

            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                T2 indexOffset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + highIndexOffset);
                AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask5, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    hFullBatchCount);
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(initial2DRegGrad, helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegGradOne, helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t));

        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                T2 indexOffset =
                    (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxActual + highIndexOffset);
                AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask6, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    wFullBatchCount);
            }

            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount +
                     (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                T2 indexOffset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount +
                     (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxActual + highIndexOffset);
                AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask7, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    1);
            }
        }
    }
}

} // namespace MaxPoolGradNCHWBackwardBaseNameSpace
#endif // MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_
