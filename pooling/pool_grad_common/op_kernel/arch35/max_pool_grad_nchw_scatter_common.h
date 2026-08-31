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
 * \file max_pool_grad_nchw_scatter_common.h
 * \brief NCHW格式MaxPoolGrad通用Kernel实现
 */

#ifndef MAX_POOL_GRAD_NCHW_SCATTER_COMMON_H_
#define MAX_POOL_GRAD_NCHW_SCATTER_COMMON_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_grad_with_argmax_base_common.h"
#include "../pool_3d_common/arch35/pool_3d_common.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"
#include "pool_grad_index_common.h"

namespace MaxPoolGradNCHWNameSpace {
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;

template <typename T, const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchw(Reg::RegTensor<T>& argmaxReg, Reg::RegTensor<int32_t>& hIndexReg,
                                     Reg::RegTensor<int32_t>& wIndexReg, Reg::RegTensor<T>& wOutputConstReg,
                                     int64_t curHIndex, int64_t curWIndex, int32_t wOutputAligned,
                                     int32_t highOutputOffset, int32_t highOutputPlaneActual,
                                     int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<T> hTmpIndexReg;
    AscendC::Reg::RegTensor<T> wTmpIndexReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(hTmpIndexReg, argmaxReg, wOutputConstReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, hTmpIndexReg, T(-curHIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(hIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)hIndexReg, (AscendC::Reg::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::Reg::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::Reg::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::Reg::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Reg::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)wIndexReg, (AscendC::Reg::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::Reg::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, hIndexReg, T(wOutputAligned), allMaskU32);
    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      wIndexReg, allMaskU32);

    AscendC::Reg::Adds((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       highOutputOffset, allMaskU32);

    if constexpr (IS_MUL_NC == 1) {
        AscendC::Reg::RegTensor<int32_t> highIncReg;
        AscendC::Reg::Arange(highIncReg, 0);
        AscendC::Reg::RegTensor<int32_t> constReg;
        AscendC::Reg::Duplicate(constReg, highArgmaxPlaneActual);
        AscendC::Reg::Div(highIncReg, highIncReg, constReg, allMaskU32);
        AscendC::Reg::Muls(highIncReg, highIncReg, highOutputPlaneActual, allMaskU32);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                          highIncReg, allMaskU32);
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                      __local_mem__ T2* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                      Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                      Reg::RegTensor<T3>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex,
                                      int32_t wOutputAligned, int32_t highOutputOffset,
                                      Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg,
                                      Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    // 相对索引
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                                   pregT2);
    IndexConvNchw<T3>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                      highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulNCNchw(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                   __local_mem__ T2* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                   Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                   Reg::RegTensor<T3>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex,
                                   int32_t wOutputAligned, int32_t highOutputOffset,
                                   Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg,
                                   Reg::RegTensor<int32_t>& hMaxReg, int32_t highOutputPlaneActual,
                                   int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<T3> argmaxReg;
    // 相对索引
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2, T3>(maskT2);
    GetConCurrentInput<T1, T2, T3>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                                   pregT2);
    IndexConvNchw<T3, 1>(argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                         highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

    AscendC::Reg::Duplicate(constReg, T(fullBatchColNum));
    AscendC::Reg::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, T(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, T(1 * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, T(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rowNumCount * colNumAligned), preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, T(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE = 0>
class MaxPoolGradKernelNCHWBase {
public:
    __aicore__ inline MaxPoolGradKernelNCHWBase(void){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
                                const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
    __aicore__ inline void ParseTilingData(const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
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

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ?
                                                             BLOCK_SIZE / sizeof(T1) :
                                                             BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::ParseTilingData(
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

    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;

    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
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
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    highAxisIndex_ = baseBlockIdx / (hOutputOuter_ * wOutputOuter_);
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;

    int64_t tempTail = baseBlockIdx % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempTail / wOutputOuter_;
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;

    wAxisIndex_ = tempTail % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;
    wOutputAligned_ = (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    int64_t hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    int64_t wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    int64_t wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;
    wArgmaxAligned_ = (wArgmaxActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;

    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;

    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::Process()
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
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::Compute()
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
        multipleLineProcessVF1(yAddr, gradAddr, argmaxAddr); // HW 并发处理
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
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::ProcessPerLoop()
{
    if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0) {
        ProcessNoArgmaxBlock(); // ceilMode为false时，最后的尾块可能是这种情况
        return;
    }

    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::CopyIn()
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
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hArgmaxActual_),
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
    DataCopyExtParams copyOutParamT2 = {static_cast<uint16_t>(hArgmaxActual_),
                                        static_cast<uint32_t>(wArgmaxActual_ * sizeof(T2)),
                                        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T2)),
                                        static_cast<uint32_t>(dstStrideT2), static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::CopyOut()
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

// Scatter处理实现

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::singleLineProcessVF(
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
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        PoolGradCommon::GenInitial1DIndices((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                           hIdx * wArgmaxAligned + highArgmaxOffset);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, all, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
                // 尾段整batch  用不满mask
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                               parallelRegIndex, wRemainBatchCount, wOutputConstReg,
                                                               curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                               zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                 repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, one, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF1(
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
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        PoolGradCommon::GenInitial2DIndices((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize,
                                            hProBatchSize, wArgmaxAligned, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize,
                                      wArgmaxAligned);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    // 整batch
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                     hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, maskBlock, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }

                    // 尾段零散点
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                     hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, blockOne, wOutputConstReg,
                            curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
            }

            // 尾行  完整hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                               parallelRegIndex, maskRemainBatch, wOutputConstReg,
                                                               curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                               zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                               parallelRegIndex, remainBatchOne, wOutputConstReg,
                                                               curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                               zeroConstReg, wMaxReg, hMaxReg);
                }
            }
            // 尾行  零散hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                 hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                               parallelRegIndex, maskRemainTail, wOutputConstReg,
                                                               curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                               zeroConstReg, wMaxReg, hMaxReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex, remainTailOne, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2(
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
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        GenInitial3DIndices((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, hArgmaxActual);

        PoolGradCommon::GenInitial2DIndices((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize,
                                            hArgmaxActual, wArgmaxAligned, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual,
                                      wArgmaxAligned);

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask0, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, whFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask1, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, hFullBatchCount);
                }
            }

            // hRemainTail
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask2, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, wFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                 (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask3, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, 1);
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
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask4, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, whFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                             highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask5, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, hFullBatchCount);
            }
        }

        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned +
                             highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask6, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, wFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                             (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask7, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, 1);
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradKernelNCHWBase<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2Int64(
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
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;

        GenInitial3DIndices((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, hArgmaxActual);

        PoolGradCommon::GenInitial2DIndices((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize,
                                            hArgmaxActual, wArgmaxAligned, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual,
                                      wArgmaxAligned);

        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(helpAddr, initial3DRegIndex, allMask);
        AscendC::Reg::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);
        AscendC::Reg::DataCopy(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);
        AscendC::Reg::DataCopy(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMask);
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::Reg::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask0, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, whFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask1, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, hFullBatchCount);
                }
            }

            // hRemainTail
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                 highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask2, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, wFullBatchCount);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                 (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                                            parallelRegIndex, mask3, wOutputConstReg, curHIndex,
                                                            curWIndex, wOutputAligned, highOutputOffset, zeroConstReg,
                                                            wMaxReg, hMaxReg, highOutputPlaneActual, 1);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::Reg::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        // highBlockRemainTail
        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // 整H batch
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask4, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, whFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                             highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask5, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, hFullBatchCount);
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, T2(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<T3> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, T3(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::DataCopy(initial3DRegIndex, helpAddr);
        AscendC::Reg::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        // highBlockRemainTail
        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned +
                             highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask6, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, wFullBatchCount);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                             (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                DoMulNCNchw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegIndex,
                                                        mask7, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                                        highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                        highOutputPlaneActual, 1);
            }
        }
    }
}

template <const uint32_t IS_MUL_NC = 0>
__aicore__ inline void IndexConvNchwFastDiv(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<int32_t>& hIndexReg,
                                            Reg::RegTensor<int32_t>& wIndexReg, Reg::RegTensor<uint32_t>& magicReg,
                                            int16_t shift, int64_t curHIndex, int64_t curWIndex, int32_t wOutput,
                                            int32_t wOutputAligned, int32_t highOutputOffset,
                                            int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    Reg::MaskReg allMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> hTmpU32;
    Reg::RegTensor<uint32_t> wTmpU32;
    Reg::RegTensor<int32_t> highIncReg;
    Reg::RegTensor<uint32_t> magicHighReg;
    Reg::RegTensor<uint32_t> highIncU32;

    PoolUtils::Compute::FastDivImpl(hTmpU32, (Reg::RegTensor<uint32_t>&)argmaxReg, magicReg, shift, allMask);

    Reg::Adds(hIndexReg, (Reg::RegTensor<int32_t>&)hTmpU32, int32_t(-curHIndex), allMask);

    Reg::Muls(wTmpU32, hTmpU32, uint32_t(wOutput), allMask);
    Reg::Sub(wTmpU32, (Reg::RegTensor<uint32_t>&)argmaxReg, wTmpU32, allMask);

    Reg::Adds(wIndexReg, (Reg::RegTensor<int32_t>&)wTmpU32, int32_t(-curWIndex), allMask);

    Reg::Muls(argmaxReg, hIndexReg, int32_t(wOutputAligned), allMask);
    Reg::Add(argmaxReg, argmaxReg, wIndexReg, allMask);
    Reg::Adds(argmaxReg, argmaxReg, highOutputOffset, allMask);

    if constexpr (IS_MUL_NC == 1) {
        Reg::RegTensor<int32_t> highIncReg;
        Reg::Arange(highIncReg, 0);
        Reg::RegTensor<uint32_t> magicHighReg;
        uint32_t magicHigh = 0;
        uint32_t shiftHigh = 0;
        GetUintDivMagicAndShift<uint32_t>(magicHigh, shiftHigh, static_cast<uint32_t>(highArgmaxPlaneActual));
        Reg::Duplicate(magicHighReg, magicHigh);
        Reg::RegTensor<uint32_t> highIncU32;
        PoolUtils::Compute::FastDivImpl(highIncU32, (Reg::RegTensor<uint32_t>&)highIncReg, magicHighReg,
                                        static_cast<int16_t>(shiftHigh), allMask);
        Reg::Muls(highIncReg, (Reg::RegTensor<int32_t>&)highIncU32, highOutputPlaneActual, allMask);
        Reg::Add(argmaxReg, argmaxReg, highIncReg, allMask);
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE, const bool IS_OVERLAP>
__aicore__ inline void DoSingleNCNchwFastDiv(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                             __local_mem__ int32_t* argmaxAddr,
                                             Reg::RegTensor<uint32_t>& parallelRegIndex,
                                             Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                             Reg::RegTensor<uint32_t>& magicReg, int16_t shift, int64_t curHIndex,
                                             int64_t curWIndex, int32_t wOutput, int32_t wOutputAligned,
                                             int32_t highOutputOffset, Reg::RegTensor<int32_t>& zeroConstReg,
                                             Reg::RegTensor<int32_t>& wMaxReg, Reg::RegTensor<int32_t>& hMaxReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<int32_t, int32_t>(maskT2);
    GetConCurrentInput<T1, int32_t, int32_t>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex,
                                             parallelRegGrad, pregT1, pregT2);
    IndexConvNchwFastDiv<0>(argmaxReg, hIndexReg, wIndexReg, magicReg, shift, curHIndex, curWIndex, wOutput,
                            wOutputAligned, highOutputOffset, 0, 0);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }
    if constexpr (IS_OVERLAP) {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }

    GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, const uint32_t IS_CHECK_RANGE, const bool IS_OVERLAP>
__aicore__ inline void DoMulNCNchwFastDiv(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                          __local_mem__ int32_t* argmaxAddr, Reg::RegTensor<uint32_t>& parallelRegIndex,
                                          Reg::RegTensor<uint32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                          Reg::RegTensor<uint32_t>& magicReg, int16_t shift, int64_t curHIndex,
                                          int64_t curWIndex, int32_t wOutput, int32_t wOutputAligned,
                                          int32_t highOutputOffset, Reg::RegTensor<int32_t>& zeroConstReg,
                                          Reg::RegTensor<int32_t>& wMaxReg, Reg::RegTensor<int32_t>& hMaxReg,
                                          int32_t highOutputPlaneActual, int32_t highArgmaxPlaneActual)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<int32_t, int32_t>(maskT2);
    GetConCurrentInput<T1, int32_t, int32_t>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex,
                                             parallelRegGrad, pregT1, pregT2);
    IndexConvNchwFastDiv<1>(argmaxReg, hIndexReg, wIndexReg, magicReg, shift, curHIndex, curWIndex, wOutput,
                            wOutputAligned, highOutputOffset, highOutputPlaneActual, highArgmaxPlaneActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }
    if constexpr (IS_OVERLAP) {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }

    GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T>
__simd_callee__ inline void GenInitial1DIndicesVF(Reg::RegTensor<T>& indexReg, int64_t colGenRate)
{
    Reg::Arange(indexReg, 0);
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Muls(indexReg, indexReg, T(colGenRate), preg);
}

__simd_callee__ inline void FastDivImplVF(Reg::RegTensor<uint32_t>& res, Reg::RegTensor<uint32_t>& src,
                                          Reg::RegTensor<uint32_t>& magic, int16_t shift, Reg::MaskReg& mask)
{
    Reg::RegTensor<uint32_t> tmp;
    Reg::Mull(tmp, res, src, magic, mask);
    Reg::Add(tmp, src, res, mask);
    Reg::ShiftRights(res, tmp, shift, mask);
}

template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexInt32FastDiv(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left,
                                                int32_t wInputActualNoPad, int32_t hIndexBase,
                                                Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, uint32_t magic,
                                                uint32_t shift)
{
    Reg::RegTensor<int32_t> hIndexReg;
    Reg::RegTensor<int32_t> wIndexReg;
    Reg::RegTensor<int32_t> zeroReg;
    Reg::RegTensor<uint32_t> divResultU32;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::MaskReg negInfMask;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    Reg::Duplicate(magicReg, magic);
    Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    PoolUtils::Compute::FastDivImpl(divResultU32, (Reg::RegTensor<uint32_t>&)srcReg, magicReg,
                                    static_cast<int16_t>(shift), allMaskB32);

    Reg::Adds(hIndexReg, (Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    Reg::Sub((Reg::RegTensor<uint32_t>&)srcReg, (Reg::RegTensor<uint32_t>&)srcReg, divResultU32, allMaskB32);
    Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

template <const uint32_t IS_PAD>
__simd_callee__ inline void ConvertIndexInt32FastDivVF(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                       int32_t left, int32_t wInputActualNoPad, int32_t hIndexBase,
                                                       Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset,
                                                       uint32_t magic, uint32_t shift)
{
    Reg::RegTensor<int32_t> hIndexReg;
    Reg::RegTensor<int32_t> wIndexReg;
    Reg::RegTensor<int32_t> zeroReg;
    Reg::RegTensor<uint32_t> divResultU32;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::MaskReg negInfMask;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    Reg::Duplicate(magicReg, magic);
    Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    FastDivImplVF(divResultU32, (Reg::RegTensor<uint32_t>&)srcReg, magicReg, static_cast<int16_t>(shift), allMaskB32);

    Reg::Adds(hIndexReg, (Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    Reg::Sub((Reg::RegTensor<uint32_t>&)srcReg, (Reg::RegTensor<uint32_t>&)srcReg, divResultU32, allMaskB32);
    Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

template <const uint32_t IS_PAD>
__simd_callee__ inline void ConvertIndexNcInt32FastDivVF(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                         int32_t left, int32_t wInputActualNoPad, int32_t hIndexBase,
                                                         Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset,
                                                         int32_t ncOutputCount, int32_t inputNcSize, uint32_t magicNc,
                                                         uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride)
{
    Reg::RegTensor<int32_t> ncIndexReg;
    Reg::RegTensor<uint32_t> divResultU32;
    Reg::RegTensor<uint32_t> magicNcReg;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(magicNcReg, magicNc);
    Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    FastDivImplVF(divResultU32, (Reg::RegTensor<uint32_t>&)ncIndexReg, magicNcReg, static_cast<int16_t>(shiftNc),
                  allMaskB32);
    Reg::Muls(ncIndexReg, (Reg::RegTensor<int32_t>&)divResultU32, inputNcSize, allMaskB32);
    Reg::Sub(srcReg, srcReg, ncIndexReg, allMaskB32);

    ConvertIndexInt32FastDivVF<IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                       ncInputOffset, magicWStride, shiftWStride);
}

} // namespace MaxPoolGradNCHWNameSpace
#endif // MAX_POOL_GRAD_NCHW_SCATTER_COMMON_H_
