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
 * \file max_pool_grad_with_argmax_v3_nhwc_kernel.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_NHWC_KERNEL_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_NHWC_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool_grad_with_argmax_v3_base.h"

namespace MaxPoolGradWithArgmaxV3NHWCNameSpace {
// preg输入为T2 输出为T1
template <typename T, const uint32_t IS_MUL_C = 0>
__aicore__ inline void IndexConvNhwc(
    MicroAPI::RegTensor<T>& argmaxReg, MicroAPI::RegTensor<int32_t>& hIndexReg, MicroAPI::RegTensor<int32_t>& wIndexReg,
    MicroAPI::RegTensor<T>& wOutputConstReg, int64_t curHIndex, int64_t curWIndex, int32_t wOutputActual,
    int32_t cOutputAligned, int32_t cOffset, int32_t nOffset, int32_t cOutputActual)
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
        AscendC::MicroAPI::Pack(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)hIndexReg, (AscendC::MicroAPI::RegTensor<int64_t>&)hIndexReg);
    } else {
        AscendC::MicroAPI::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);
    }

    AscendC::MicroAPI::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMask);
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::MicroAPI::Adds(tmpReg, wTmpIndexReg, T(-curWIndex), allMask);
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(wIndexReg, tmpReg, allMask);
        AscendC::MicroAPI::Pack(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)wIndexReg, (AscendC::MicroAPI::RegTensor<int64_t>&)wIndexReg);
    } else {
        AscendC::MicroAPI::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);
    }

    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, hIndexReg, wOutputActual, allMaskU32);
    AscendC::MicroAPI::Add(
        (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, wIndexReg,
        allMaskU32); // H + W

    AscendC::MicroAPI::Muls(
        (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
        cOutputAligned, allMaskU32);
    AscendC::MicroAPI::RegTensor<int32_t> cIncReg;
    AscendC::MicroAPI::Arange(cIncReg, (cOffset));
    if constexpr (IS_MUL_C == 1) {
        AscendC::MicroAPI::RegTensor<int32_t> constReg;
        AscendC::MicroAPI::Duplicate(constReg, cOutputActual);
        AscendC::MicroAPI::RegTensor<int32_t> tmpReg;
        AscendC::MicroAPI::Div(tmpReg, cIncReg, constReg, allMaskU32);
        AscendC::MicroAPI::Mul(tmpReg, tmpReg, constReg, allMaskU32);
        AscendC::MicroAPI::Sub(cIncReg, cIncReg, tmpReg, allMaskU32);
    }

    AscendC::MicroAPI::Add(
        (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, cIncReg,
        allMaskU32); // H + W + C
    AscendC::MicroAPI::Adds(
        (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, nOffset,
        allMaskU32); // H + W + C + N
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void GetContinuousInput(
    MicroAPI::RegTensor<T3>& argmaxReg, MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
    __local_mem__ T2* argmaxAddr, uint32_t argmaxOffset)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(gradRegT1, gradAddr + argmaxOffset);
        AscendC::MicroAPI::UnPack(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1, (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopy(gradReg, gradAddr + argmaxOffset);
    }

    if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int32_t>::value) {
        AscendC::MicroAPI::DataCopy(argmaxReg, argmaxAddr + argmaxOffset);
    } else if constexpr (std::is_same<T3, int32_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::MicroAPI::RegTensor<T2, AscendC::MicroAPI::RegTraitNumTwo> argmaxRegTwo;
        AscendC::MicroAPI::DataCopy(argmaxRegTwo, argmaxAddr + argmaxOffset);
        argmaxReg = (AscendC::MicroAPI::RegTensor<T3>&)argmaxRegTwo.reg[0];
    } else if constexpr (std::is_same<T3, int64_t>::value && std::is_same<T2, int64_t>::value) {
        AscendC::MicroAPI::DataCopy(argmaxReg, argmaxAddr + argmaxOffset);
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleCNhwc(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr, uint32_t argmaxOffset,
    uint32_t argmaxMaskCount, int64_t curHIndex, int64_t curWIndex, int32_t wOutputActual, int32_t cOutputAligned,
    int32_t cOffset, int32_t nOffset, int32_t cOutputActual, MicroAPI::RegTensor<int32_t>& zeroConstReg,
    MicroAPI::RegTensor<int32_t>& wMaxReg, MicroAPI::RegTensor<int32_t>& hMaxReg,
    MicroAPI::RegTensor<T3>& wOutputConstReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<T3> argmaxReg;
    // 相对索引
    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;

    GetContinuousInput(argmaxReg, gradReg, gradAddr, argmaxAddr, argmaxOffset);
    IndexConvNhwc<T3, 0>(
        argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputActual, cOutputAligned, cOffset,
        nOffset, cOutputActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulCNhwc(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount, int64_t curHIndex, int64_t curWIndex,
    int32_t wOutputActual, int32_t cOutputAligned, int32_t cOffset, int32_t nOffset, int32_t cOutputActual,
    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& wMaxReg,
    MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<T3>& wOutputConstReg)
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
    IndexConvNhwc<T3, 1>(
        argmaxReg, hIndexReg, wIndexReg, wOutputConstReg, curHIndex, curWIndex, wOutputActual, cOutputAligned, cOffset,
        nOffset, cOutputActual);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    if constexpr (IS_CHECK_RANGE == 1) {
        FilterMask(pregArgmax, hIndexReg, wIndexReg, zeroConstReg, wMaxReg, hMaxReg);
    }

    GradientAcc<T3>(yAddr, gradReg, argmaxReg, pregArgmax);
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
    AscendC::MicroAPI::Duplicate(constReg, T(fullBatchColNum * cOutputActual));

    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(fullBatchColNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(cOutputActual));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);

    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, T(cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);

    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(colGenRate * cOutputAligned), preg);

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

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE = 0>
class MaxPoolGradWithArgmaxV3KernelNHWC {
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3KernelNHWC(void){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
        const MaxPoolGradWithArgmaxV3NHWCTilingData& tilingData);
    __aicore__ inline void ParseTilingData(const MaxPoolGradWithArgmaxV3NHWCTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void ConCProcVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ConCMergeWProcVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ConCMergeHWProcVF(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void ConCMergeHWProcVFInt64(
        __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
        __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> gradGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;

    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;

    int64_t cOutput_ = 1;
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

    int64_t nOutputInner_ = 1;
    int64_t nOutputTail_ = 1;
    int64_t nOutputOuter_ = 1;
    int64_t nOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;

    int64_t cOutputInner_ = 1;
    int64_t cOutputTail_ = 1;
    int64_t cOutputOuter_ = 1;
    int64_t cOutputActual_ = 1;
    int64_t cOutputAligned_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;

    int64_t nAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;

    int64_t nOutputArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;
    int64_t cAxisArgmaxOffset_ = 0;

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
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPoolGradWithArgmaxV3NHWCTilingData& tilingData)
{
    hArgmax_ = tilingData.hArgmax;
    wArgmax_ = tilingData.wArgmax;

    cOutput_ = tilingData.cOutput;
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

    nOutputInner_ = tilingData.nOutputInner;
    nOutputTail_ = tilingData.nOutputTail;
    nOutputOuter_ = tilingData.nOutputOuter;

    hOutputInner_ = tilingData.hOutputInner;
    hOutputTail_ = tilingData.hOutputTail;
    hOutputOuter_ = tilingData.hOutputOuter;

    wOutputInner_ = tilingData.wOutputInner;
    wOutputTail_ = tilingData.wOutputTail;
    wOutputOuter_ = tilingData.wOutputOuter;

    cOutputInner_ = tilingData.cOutputInner;
    cOutputTail_ = tilingData.cOutputTail;
    cOutputOuter_ = tilingData.cOutputOuter;

    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    usedCoreNum_ = tilingData.usedCoreNum;

    outputBufferSize_ = tilingData.outputBufferSize;
    gradBufferSize_ = tilingData.gradBufferSize;
    argmaxBufferSize_ = tilingData.argmaxBufferSize;

    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
    curHProBatchSize_ = hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPoolGradWithArgmaxV3NHWCTilingData& tilingData)
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
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    nAxisIndex_ = baseBlockIdx / (hOutputOuter_ * wOutputOuter_ * cOutputOuter_);
    nOutputActual_ = nAxisIndex_ == (nOutputOuter_ - 1) ? nOutputTail_ : nOutputInner_;

    int64_t tempNTail = baseBlockIdx % (hOutputOuter_ * wOutputOuter_ * cOutputOuter_);
    cAxisIndex_ = tempNTail / (hOutputOuter_ * wOutputOuter_);
    cOutputActual_ = cAxisIndex_ == (cOutputOuter_ - 1) ? cOutputTail_ : cOutputInner_;
    cOutputAligned_ =
        (cOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t tempCTail = tempNTail % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempCTail / (wOutputOuter_);
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;

    wAxisIndex_ = tempCTail % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;

    int64_t hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    int64_t hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    int64_t wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    int64_t wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;

    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;

    nOutputArgmaxOffset_ = nAxisIndex_ * nOutputInner_ * argmaxPlaneSize_ * cOutput_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_ * cOutput_;
    wAxisArgmaxOffset_ = wArgmaxActualStart * cOutput_;
    cAxisArgmaxOffset_ = cAxisIndex_ * cOutputInner_;
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::Process()
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
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::Compute()
{
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t concurrencyCount = computeSizeT2 / cOutputActual_;
    if (concurrencyCount < 2) {
        ConCProcVF(yAddr, gradAddr, argmaxAddr);
    } else {
        uint32_t wFullBatchCount = wArgmaxActual_ / curWProBatchSize_;
        uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;
        if (hConcurrentCount < 2) {
            ConCMergeWProcVF(yAddr, gradAddr, argmaxAddr);
        } else {
            LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
            __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
            if constexpr (std::is_same<T3, int64_t>::value) {
                ConCMergeHWProcVFInt64(yAddr, gradAddr, argmaxAddr, helpAddr);
            } else {
                ConCMergeHWProcVF(yAddr, gradAddr, argmaxAddr, helpAddr);
            }
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
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ProcessPerLoop()
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
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();

    int64_t argmaxGmOffset = nOutputArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_ + cAxisArgmaxOffset_;

    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hArgmaxActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = wArgmax_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = argmaxPlaneSize_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = wArgmaxActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = hArgmaxActual_ * wArgmaxActual_ * cOutputAligned_ * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(wArgmaxActual_), static_cast<uint32_t>(cOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);

    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT2;
    loopModeParamsT2.loop1Size = hArgmaxActual_;
    loopModeParamsT2.loop2Size = nOutputActual_;
    loopModeParamsT2.loop1SrcStride = wArgmax_ * cOutput_ * sizeof(T2);
    loopModeParamsT2.loop2SrcStride = argmaxPlaneSize_ * cOutput_ * sizeof(T2);
    loopModeParamsT2.loop1DstStride = wArgmaxActual_ * cOutputAligned_ * sizeof(T2);
    loopModeParamsT2.loop2DstStride = hArgmaxActual_ * wArgmaxActual_ * cOutputAligned_ * sizeof(T2);

    uint32_t dstStrideT2 = (cOutputAligned_ - cOutputActual_) * sizeof(T2) / BLOCK_SIZE;
    SetLoopModePara(loopModeParamsT2, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(wArgmaxActual_), static_cast<uint32_t>(cOutputActual_ * sizeof(T2)),
        static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T2)), static_cast<uint32_t>(dstStrideT2),
        static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ConCProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutput = wOutput_;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;

    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;

    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wArgmaxActual = wArgmaxActual_;
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t cOutputActual = cOutputActual_;

    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t cRepeatimes = cOutputActual / computeSizeT2;
    uint16_t cRemain = cOutputActual - cRepeatimes * computeSizeT2;
    uint16_t cRemainLoopTimes = cRemain == 0 ? 0 : 1;
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

        for (uint16_t cIdx = 0; cIdx < cRepeatimes; ++cIdx) {
            uint32_t cOffset = cIdx * computeSizeT2;
            for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
                uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                    for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                        uint32_t argmaxOffset =
                            ((nIdx * hArgmaxActual + hIdx) * wArgmaxActual + wIdx) * cOutputAligned + cOffset;
                        DoSingleCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, argmaxOffset, computeSizeT2, curHIndex, curWIndex,
                            wOutputActual, cOutputAligned, cOffset, nOffset, cOutputActual, zeroConstReg, wMaxReg,
                            hMaxReg, wOutputConstReg);
                    }
                }
            }
        }

        // cRemain
        for (uint16_t cIdx = 0; cIdx < cRemainLoopTimes; ++cIdx) {
            uint32_t cOffset = cRepeatimes * computeSizeT2;
            for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
                uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
                    for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                        uint32_t argmaxOffset =
                            ((nIdx * hArgmaxActual + hIdx) * wArgmaxActual + wIdx) * cOutputAligned + cOffset;
                        DoSingleCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, argmaxOffset, cRemain, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, cOffset, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg,
                            wOutputConstReg);
                    }
                }
            }
        }
    }
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ConCMergeWProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutput = wOutput_;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;

    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;

    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t concurrencyCount = computeSizeT2 / cOutputActual;

    uint16_t repeatimes = wFullBatchCount / concurrencyCount;
    uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * concurrencyCount;
    uint32_t wRemainBatch = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;

    uint32_t mask0 = concurrencyCount * cOutputActual;
    uint32_t mask1 = wRemainBatch * cOutputActual;
    uint32_t mask2 = 1 * cOutputActual;

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
        GenInitial3DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
            wFullBatchCount, cOutputActual, cOutputAligned);

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                        cOutputAligned +
                                    nArgmaxOffset;

                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }
                }
                // 尾段整batch  用不满mask
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wRemainBatch * wProBatchSize +
                                 repeatimes * concurrencyCount * wProBatchSize + hIdx * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ConCMergeHWProcVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;

    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;

    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wArgmaxActual = wArgmaxActual_;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;

    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t concurrencyCount = computeSizeT2 / cOutputActual;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    int64_t wOutput = wOutput_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;

    uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t mask0 = wFullBatchCount * hConcurrentCount * cOutputActual;
    uint32_t mask1 = 1 * hConcurrentCount * cOutputActual;
    uint32_t mask2 = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t mask3 = 1 * hRemainBatchCount * cOutputActual;
    uint32_t mask4 = wFullBatchCount * cOutputActual;
    uint32_t mask5 = 1 * cOutputActual;

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

        GenInitial3DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
            wFullBatchCount, cOutputActual, cOutputAligned);
        Gen3DIndexOne(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual, cOutputActual,
            cOutputAligned);

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    // 整batch
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                     hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                        cOutputAligned +
                                    nArgmaxOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }

                    // 尾段零散点
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                     hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                        cOutputAligned +
                                    nArgmaxOffset;

                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }
                }
            }

            // 尾行  完整hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }
            }
            // 尾行  零散hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxActual + hRemainBatchCount * hProBatchSize * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                            cOutputAligned +
                        nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                 hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nArgmaxOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::ConCMergeHWProcVFInt64(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int64_t hOutputActual = hOutputActual_;
    int64_t wOutputActual = wOutputActual_;

    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;

    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wArgmaxActual = wArgmaxActual_;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;

    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t concurrencyCount = computeSizeT2 / cOutputActual;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    int64_t wOutput = wOutput_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;

    uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t mask0 = wFullBatchCount * hConcurrentCount * cOutputActual;
    uint32_t mask1 = 1 * hConcurrentCount * cOutputActual;
    uint32_t mask2 = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t mask3 = 1 * hRemainBatchCount * cOutputActual;
    uint32_t mask4 = wFullBatchCount * cOutputActual;
    uint32_t mask5 = 1 * cOutputActual;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        GenInitial3DIndices(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize, wArgmaxActual,
            wFullBatchCount, cOutputActual, cOutputAligned);
        Gen3DIndexOne(
            (AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual, cOutputActual,
            cOutputAligned);

        AscendC::MicroAPI::MaskReg allMask =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initialRegIndexOne, allMask);
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

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
        AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    // 整batch
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                     hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                        cOutputAligned +
                                    nOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }

                    // 尾段零散点
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                     hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                        cOutputAligned +
                                    nOffset;

                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, curHIndex, curWIndex, wOutputActual,
                            cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                    }
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

        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
        AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
            uint32_t nOffset = nIdx * hOutputActual * wOutputActual * cOutputAligned;
            // 尾行  完整hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }
            }
            // 尾行  零散hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset =
                        (wBatchIdx + hProBatchIdx * wArgmaxActual + hRemainBatchCount * hProBatchSize * wArgmaxActual +
                         blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                            cOutputAligned +
                        nOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                 hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                 blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                    cOutputAligned +
                                nOffset;

                    AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                    DoMulCNhwc<T1, T2, T3, IS_CHECK_RANGE>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5, curHIndex, curWIndex, wOutputActual,
                        cOutputAligned, 0, nOffset, cOutputActual, zeroConstReg, wMaxReg, hMaxReg, wOutputConstReg);
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradWithArgmaxV3KernelNHWC<T1, T2, T3, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();

    int64_t outputPlaneSize = hOutput_ * wOutput_ * cOutput_;
    int64_t nOutputAxisOffset = nAxisIndex_ * nOutputInner_ * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * hOutputInner_ * wOutput_ * cOutput_;
    int64_t wOutputAxisOffset = wAxisIndex_ * wOutputInner_ * cOutput_;
    int64_t cOutputAxisOffset = cAxisIndex_ * cOutputInner_;
    int64_t outputGmOffset = nOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset + cOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hOutputActual_;
    loopModeParamsT1.loop2Size = nOutputActual_;
    loopModeParamsT1.loop1SrcStride = wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = hOutputActual_ * wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = wOutput_ * cOutput_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = outputPlaneSize * sizeof(T1);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(wOutputActual_), static_cast<uint32_t>(cOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>((cOutput_ - cOutputActual_) * sizeof(T1)),
        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}
} // namespace MaxPoolGradWithArgmaxV3NHWCNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX_V3_NHWC_KERNEL_H_
