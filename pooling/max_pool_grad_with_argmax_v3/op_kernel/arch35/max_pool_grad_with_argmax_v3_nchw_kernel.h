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
 * \file max_pool_grad_with_argmax_v3_nchw_kernel.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_base_common.h"
#include "../max_pool_grad_with_argmax_common/max_pool_grad_with_argmax_struct_common.h"

namespace MaxPoolGradWithArgmaxV3NCHWNameSpace
{

// argmaxReg输入为T3 输出为int32_t
template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchw(MicroAPI::RegTensor<T>& argmaxReg, MicroAPI::RegTensor<int32_t>& hIndexReg,
                                     MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<T>& wOutputConstReg,
                                     int64_t curHIndex, int64_t curWIndex, int32_t wOutputAligned,
                                     int32_t highOutputOffset, int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::MicroAPI::RegTensor<T> hTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> wTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg allMaskU32 =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Div(hTmpIndexReg, argmaxReg, wOutputConstReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)hIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::MicroAPI::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::MicroAPI::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)wIndexReg,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::MicroAPI::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, hIndexReg, T(wOutputAligned),
                            allMaskU32);
    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, wIndexReg, allMaskU32);

    AscendC::MicroAPI::Adds((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                            (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highOutputOffset, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> highIncReg;
        AscendC::MicroAPI::Arange(highIncReg, 0);
        AscendC::MicroAPI::RegTensor<int32_t> constReg;
        AscendC::MicroAPI::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::MicroAPI::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::MicroAPI::Muls(highIncReg, highIncReg, highOutputPlaneActual,
                            allMaskU32);
        AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, highIncReg,
                           allMaskU32);  
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                              __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
                              uint32_t argmaxMaskCount, MicroAPI::RegTensor<T3>& wOutputConstReg, int64_t curHIndex,
                              int64_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset,
                              MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
                              MicroAPI::RegTensor<int32_t>& hMaxReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg;
    // 相对索引
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);
    IndexConvNchw<T3>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                      highOutputOffset,0 ,0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                              __local_mem__ T2* argmaxAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
                              uint32_t argmaxMaskCount, MicroAPI::RegTensor<T3>& wOutputConstReg, int64_t curHIndex,
                              int64_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset,
                              MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
                              MicroAPI::RegTensor<int32_t>& hMaxReg, int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg;
    // 相对索引
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);
    IndexConvNchw<T3, 1>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                      highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__aicore__ inline void GenInitial1DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(colGenRate), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNumAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t rowNumCount)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
class MaxPoolGradWithArgmaxV3NCHWKernel
{
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3NCHWKernel(void){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
                                const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
    __aicore__ inline void ParseTilingData(const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void singleLineProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                               __local_mem__ T2* argmaxAddr);
    __aicore__ inline void multipleLineProcessVF1(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                  __local_mem__ T2* argmaxAddr);
    __aicore__ inline void multipleLineProcessVF2(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                  __local_mem__ T2* argmaxAddr, __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void multipleLineProcessVF2Int64(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                       __local_mem__ T2* argmaxAddr, __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

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

    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;
    int64_t wArgmaxAligned_ = 0;

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

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ? BLOCK_SIZE / sizeof(T1) : BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
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

    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;

    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = hArgmax_ * wArgmax_;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputBufferSize_);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, gradBufferSize_);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, argmaxBufferSize_);
    pipe_.InitBuffer(helpBuf_, HELP_BUFFER);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
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

    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;

    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::Compute()
{
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    if (wConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        singleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        multipleLineProcessVF1(yAddr, gradAddr, argmaxAddr);  // HW 并发处理
    } else {
        // NCHW 并发处理
        LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
        __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
        if constexpr (std::is_same<T3, int64_t>::value) {
            multipleLineProcessVF2Int64(yAddr, gradAddr, argmaxAddr, helpAddr);
        } else {
            multipleLineProcessVF2(yAddr, gradAddr, argmaxAddr, helpAddr);
        }
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ProcessPerLoop()
{
    if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0) {
        ProcessNoArgmaxBlock();  // ceilMode为false时，最后的尾块可能是这种情况
        return;
    }

    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();

    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = argmaxPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hArgmaxActual_),
        static_cast<uint32_t>(wArgmaxActual_ * sizeof(T1)),
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);
    
    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
                             
    LoopModeParams loopModeParamsT2;
    loopModeParamsT2.loop1Size = highAxisActual_;
    loopModeParamsT2.loop2Size = 1;
    loopModeParamsT2.loop1SrcStride = argmaxPlaneSize_ * sizeof(T2);
    loopModeParamsT2.loop2SrcStride = 0;
    loopModeParamsT2.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T2);
    loopModeParamsT2.loop2DstStride = 0;

    uint32_t dstStrideT2 = (wArgmaxAligned_ - wArgmaxActual_) * sizeof(T2) / BLOCK_SIZE;
    SetLoopModePara(loopModeParamsT2, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(hArgmaxActual_),
        static_cast<uint32_t>(wArgmaxActual_ * sizeof(T2)),
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T2)),
        static_cast<uint32_t>(dstStrideT2), static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::singleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
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

    uint16_t hProBatchSize = curHProBatchSize_;
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

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                           hIdx * wArgmaxAligned + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, all, wOutputConstReg, curHIndex, curWIndex,
                            wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
                // 尾段整batch  用不满mask
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, wRemainBatchCount, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                 repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, one,
                                                               wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                               highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF1(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
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

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxAligned);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    // 整batch
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                     hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskBlock, wOutputConstReg, curHIndex,
                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }

                    // 尾段零散点
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                     hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, blockOne, wOutputConstReg, curHIndex,
                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
            }

            // 尾行  完整hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainBatch, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainBatchOne, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
            // 尾行  零散hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                         hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainTail, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                         hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainTailOne, wOutputConstReg, curHIndex,
                        curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2(
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

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, hArgmaxActual);

        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hArgmaxActual,
                            wArgmaxAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual, wArgmaxAligned);

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, whFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, hFullBatchCount);
                }
            }

            // hRemainTail
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, wFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                 (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, wOutputConstReg, curHIndex, curWIndex,
                        wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual, 1);
                }
            }
        }

        // highBlockRemainTail
        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // 整H batch
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, whFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, hFullBatchCount);
            }
        }

        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, wFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                             (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7, wOutputConstReg, curHIndex, curWIndex,
                    wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual, 1);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2Int64(
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
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;

        GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, hArgmaxActual);

        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hArgmaxActual,
                            wArgmaxAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual, wArgmaxAligned);

        AscendC::MicroAPI::MaskReg allMask =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initial3DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne,
                                    allMask);
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

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, whFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, hFullBatchCount);
                }
            }

            // hRemainTail
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2,
                                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                            highOutputPlaneActual, wFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                 (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, wOutputConstReg, curHIndex, curWIndex,
                        wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual, 1);
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

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        // highBlockRemainTail
        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // 整H batch
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, whFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, hFullBatchCount);
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

        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        // highBlockRemainTail
        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset =
                    (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6,
                                                        wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, wFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                             (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7, wOutputConstReg, curHIndex, curWIndex,
                    wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual, 1);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3NCHWKernel<T1, T2, T3, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();

    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t highOutputAxisOffset = highAxisIndex_ * highAxisInner_ * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wOutputAxisOffset = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = hOutput_ * wOutput_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}
}  // namespace MaxPoolGradWithArgmaxV3NCHWNameSpace
#endif  // MAX_POOL_GRAD_WITH_ARGMAX_V3_NCHW_KERNEL_H_
