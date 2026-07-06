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
 * \file max_pool_grad_with_argmax_simd_full_load.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMD_FULL_LOAD_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMD_FULL_LOAD_H_

#include "max_pool3d_grad_with_argmax_simd.h"
using namespace AscendC;
namespace MaxPool3DGradWithArgmaxNCDHWNameSpace {

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
class MaxPool3DGradWithArgmaxNCDHWFullLoadKernel {
public:
    __aicore__ inline MaxPool3DGradWithArgmaxNCDHWFullLoadKernel(void){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
                                const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData);
    __aicore__ inline void ParseTilingData(
        const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();

    __aicore__ inline void fullLoadSingleLineProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                       __local_mem__ T2* argmaxAddr);
    __aicore__ inline void fullLoadMultipleLineHwProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                           __local_mem__ T2* argmaxAddr);
    __aicore__ inline void fullLoadMultipleLineDhwProcessVF(__local_mem__ computeType* yAddr,
                                                            __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void fullLoadMultipleLineProcessVF2(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                          __local_mem__ T2* argmaxAddr);
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;

    int64_t dArgmax_ = 1;
    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;

    int64_t dOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t kernelD_ = 1;
    int64_t kernelH_ = 1;
    int64_t kernelW_ = 1;

    int64_t strideD_ = 1;
    int64_t strideH_ = 1;
    int64_t strideW_ = 1;

    int64_t padD_ = 0;
    int64_t padH_ = 0;
    int64_t padW_ = 0;

    int64_t dilationD_ = 1;
    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;

    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;

    int64_t dOutputInner_ = 1;
    int64_t dOutputTail_ = 1;
    int64_t dOutputOuter_ = 1;
    int64_t dOutputActual_ = 1;

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
    int64_t dAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t dArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;
    int64_t wArgmaxAligned_ = 0;
    int64_t dhwPlaneAligned_ = 0;

    int64_t highAxisArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t dAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;

    int64_t argmaxPlaneSize_ = 1;

    int64_t dProBatchSize_ = 1;
    int64_t hProBatchSize_ = 1;
    int64_t wProBatchSize_ = 1;
    int64_t curDProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;

    int64_t outPutGmOffset = 0;
    int64_t argmaxGmOffset = 0;
    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1) >= BLOCK_SIZE / sizeof(T2) ?
                                                             BLOCK_SIZE / sizeof(T1) :
                                                             BLOCK_SIZE / sizeof(T2);
    constexpr static int64_t VREG_LENGTH_DATA_NUM_T2 = platform::GetVRegSize() / sizeof(T2);
};

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::Init(
    GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, TPipe& pipeIn,
    const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = dArgmax_ * hArgmax_ * wArgmax_;
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
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPool3DGradWithArgmaxOp::MaxPool3DGradWithArgmaxNCDHWTilingData& tilingData)
{
    dArgmax_ = tilingData.dArgmax;
    hArgmax_ = tilingData.hArgmax;
    wArgmax_ = tilingData.wArgmax;

    dOutput_ = tilingData.dOutput;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;

    kernelD_ = tilingData.dKernel;
    kernelH_ = tilingData.hKernel;
    kernelW_ = tilingData.wKernel;

    strideD_ = tilingData.dStride;
    strideH_ = tilingData.hStride;
    strideW_ = tilingData.wStride;

    padD_ = tilingData.padD;
    padH_ = tilingData.padH;
    padW_ = tilingData.padW;

    dilationD_ = tilingData.dilationD;
    dilationH_ = tilingData.dilationH;
    dilationW_ = tilingData.dilationW;

    highAxisInner_ = tilingData.highAxisInner;
    highAxisTail_ = tilingData.highAxisTail;
    highAxisOuter_ = tilingData.highAxisOuter;

    dOutputInner_ = tilingData.dOutputInner;
    dOutputTail_ = tilingData.dOutputTail;
    dOutputOuter_ = tilingData.dOutputOuter;

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

    dProBatchSize_ = tilingData.dProBatchSize;
    hProBatchSize_ = tilingData.hProBatchSize;
    wProBatchSize_ = tilingData.wProBatchSize;
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::ScalarCompute(
    int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    outPutGmOffset = baseBlockIdx * highAxisInner_ * dOutput_ * hOutput_ * wOutput_;
    argmaxGmOffset = baseBlockIdx * highAxisInner_ * dArgmax_ * hArgmax_ * wArgmax_;

    highAxisIndex_ = baseBlockIdx;
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;

    dAxisIndex_ = 0;
    dOutputActual_ = dOutput_;

    hAxisIndex_ = 0;
    hOutputActual_ = hOutput_;

    wAxisIndex_ = 0;
    wOutputActual_ = wOutput_;

    wOutputAligned_ = (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t dArgmaxActualStart = 0;
    int64_t dArgmaxActualEnd = dArgmax_;
    int64_t hArgmaxActualStart = 0;
    int64_t hArgmaxActualEnd = hArgmax_;
    int64_t wArgmaxActualStart = 0;
    int64_t wArgmaxActualEnd = wArgmax_;

    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;
    wArgmaxAligned_ = (wArgmaxActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart;
    dArgmaxActual_ = dArgmaxActualEnd - dArgmaxActualStart;

    int64_t dhwPlaneElems = dArgmaxActual_ * hArgmaxActual_ * wArgmaxActual_;
    dhwPlaneAligned_ = (dhwPlaneElems + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                       MAX_DATA_NUM_IN_ONE_BLOCK;

    curDProBatchSize_ = dProBatchSize_ > dArgmaxActual_ ? dArgmaxActual_ : dProBatchSize_;
    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;
    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    dAxisArgmaxOffset_ = dArgmaxActualStart * hArgmax_ * wArgmax_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + dAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_;

    int64_t dhwPlaneElems = dArgmaxActual_ * hArgmaxActual_ * wArgmaxActual_;

    int64_t dhwPlaneBytesT1 = dhwPlaneElems * sizeof(T1);
    int64_t dhwPlaneBytesT2 = dhwPlaneElems * sizeof(T2);
    int64_t gmSrcStrideT1 = (argmaxPlaneSize_ - dhwPlaneElems) * sizeof(T1);
    int64_t gmSrcStrideT2 = (argmaxPlaneSize_ - dhwPlaneElems) * sizeof(T2);
    int64_t ubDstStrideT1 = (dhwPlaneAligned_ - dhwPlaneElems) * sizeof(T1) / BLOCK_SIZE;
    int64_t ubDstStrideT2 = (dhwPlaneAligned_ - dhwPlaneElems) * sizeof(T2) / BLOCK_SIZE;

    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(highAxisActual_), static_cast<uint32_t>(dhwPlaneBytesT1),
                                        static_cast<uint32_t>(gmSrcStrideT1), static_cast<uint32_t>(ubDstStrideT1),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);

    DataCopyPadExtParams<T2> paramsT2 = {false, 0, 0, 0};
    DataCopyExtParams copyOutParamT2 = {static_cast<uint16_t>(highAxisActual_), static_cast<uint32_t>(dhwPlaneBytesT2),
                                        static_cast<uint32_t>(gmSrcStrideT2), static_cast<uint32_t>(ubDstStrideT2),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(argmaxLocal, argmaxGm_[argmaxGmOffset], copyOutParamT2, paramsT2);

    gradQue_.EnQue(gradLocal);
    argmaxQue_.EnQue(argmaxLocal);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t outputPlaneDHW = dOutput_ * outputPlaneSize;
    int64_t ncBase = highAxisIndex_ * highAxisInner_ * outputPlaneDHW;
    int64_t dBase = dAxisIndex_ * dOutputInner_ * outputPlaneSize;
    int64_t hBase = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wBase = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = ncBase + dBase + hBase + wBase;

    DataCopyExtParams copyParam{1, static_cast<uint32_t>(highAxisActual_ * outputPlaneDHW * sizeof(T1)), 0, 0, 0};
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyParam);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::Compute()
{
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    // UB
    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();

    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    uint32_t dConcurrentCount = dArgmaxActual_ / curDProBatchSize_;

    if (wConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        fullLoadSingleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        fullLoadMultipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * dConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        fullLoadMultipleLineDhwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else {
        fullLoadMultipleLineProcessVF2(yAddr, gradAddr, argmaxAddr);
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::ProcessPerLoop()
{
    if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0 || dArgmaxActual_ <= 0) {
        ProcessNoArgmaxBlock();
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T>
__aicore__ inline void IndexConvNcNcdhwFullLoad(MicroAPI::RegTensor<T>& argmaxReg, int64_t highOutputOffset,
                                                int64_t highOutputSize, MicroAPI::RegTensor<uint32_t>& magicHighReg,
                                                int16_t shiftHigh, MicroAPI::MaskReg& pregT3,
                                                MicroAPI::RegTensor<int32_t>& indexIncReg)
{
    AscendC::MicroAPI::RegTensor<int32_t> localIndexIncReg;
    AscendC::MicroAPI::Arange(localIndexIncReg, 0);
    AscendC::MicroAPI::RegTensor<uint32_t> ncIndexIncRegU32;
    FastDivImpl(ncIndexIncRegU32, (MicroAPI::RegTensor<uint32_t>&)localIndexIncReg, magicHighReg, shiftHigh, pregT3);

    AscendC::MicroAPI::Muls((MicroAPI::RegTensor<int32_t>&)ncIndexIncRegU32,
                            (MicroAPI::RegTensor<int32_t>&)ncIndexIncRegU32, highOutputSize, pregT3);

    AscendC::MicroAPI::Adds(argmaxReg, argmaxReg, highOutputOffset, pregT3);
    AscendC::MicroAPI::Add(argmaxReg, argmaxReg, (MicroAPI::RegTensor<int32_t>&)ncIndexIncRegU32, pregT3);
}

template <typename T>
__aicore__ inline void IndexConvNcdhwFullLoad(MicroAPI::RegTensor<T>& argmaxReg, int64_t highOutputOffset,
                                              MicroAPI::MaskReg& pregT3)
{
    AscendC::MicroAPI::Adds(argmaxReg, argmaxReg, highOutputOffset, pregT3);
}

template <typename T1, typename T2>
__aicore__ inline void DoMulNCNcdhwFullLoad(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                            __local_mem__ T2* argmaxAddr,
                                            MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount,
                                            int32_t highOutputOffset, int32_t highOutputPlaneActual,
                                            int32_t highArgmaxPlaneActual, MicroAPI::RegTensor<uint32_t>& magicHighReg,
                                            int16_t shiftHigh, MicroAPI::RegTensor<int32_t>& indexIncReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> argmaxReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    uint32_t maskT3 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2>(maskT2);
    AscendC::MicroAPI::MaskReg pregT3 = AscendC::MicroAPI::UpdateMask<int32_t>(maskT3);
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);

    IndexConvNcNcdhwFullLoad<int32_t>(argmaxReg, highOutputOffset, highOutputPlaneActual, magicHighReg, shiftHigh,
                                      pregT3, indexIncReg);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);
    GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2>
__aicore__ inline void DoSingleNchwFullLoad(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                            __local_mem__ T2* argmaxAddr,
                                            MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t argmaxMaskCount,
                                            int32_t highOutputOffset)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> argmaxReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    uint32_t maskT3 = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    AscendC::MicroAPI::MaskReg pregT2 = GenT2Mask<T2>(maskT2);
    AscendC::MicroAPI::MaskReg pregT3 = AscendC::MicroAPI::UpdateMask<int32_t>(maskT3);
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, pregT1, pregT2);

    IndexConvNcdhwFullLoad<int32_t>(argmaxReg, highOutputOffset, pregT3);
    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::MicroAPI::MaskReg pregArgmax = AscendC::MicroAPI::UpdateMask<int32_t>(argmaxMask);

    GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::fullLoadSingleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    uint16_t dArgmaxActual = dArgmaxActual_;
    int64_t wArgmaxActual = wArgmaxActual_;
    int64_t wArgmaxAligned = wArgmaxActual_;
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
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * dhwPlaneAligned_;
            uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputActual;
            for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
                uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
                for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                    for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                               hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, all,
                                                         highOutputOffset);
                        }
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize +
                                           hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, wRemainBatchCount,
                                                     highOutputOffset);
                    }

                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                           repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned +
                                           dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, one,
                                                     highOutputOffset);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void
MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::fullLoadMultipleLineHwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;

    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);

    int64_t wArgmaxAligned = wArgmaxActual_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t dArgmaxActual = dArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
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

    DivMagic divW = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount));

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        DhwGenInitial2DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize,
                                   hProBatchSize, wArgmaxAligned, wFullBatchCount, divW);
        DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxAligned);
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * dhwPlaneAligned_;
            uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputActual;
            for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
                uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
                for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = (dArgmaxOffset + highArgmaxOffset + wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                         hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskBlock,
                                                         highOutputOffset);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                         hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + dArgmaxOffset +
                                         highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, blockOne,
                                                         highOutputOffset);
                        }
                    }
                }

                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                     blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                     dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainBatch,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                     blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                     dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainBatchOne,
                                                     highOutputOffset);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                     hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                     blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                     dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainTail,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                     hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                     blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                     dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainTailOne,
                                                     highOutputOffset);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void
MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::fullLoadMultipleLineDhwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    uint16_t dArgmaxActual = dArgmaxActual_;
    int64_t wArgmaxActual = wArgmaxActual_;
    int64_t wArgmaxAligned = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t dProBatchSize = curDProBatchSize_;
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint32_t hwFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t hwConcurrentCount = V_REG_SIZE / (hwFullBatchCount * sizeof(T2));

    uint16_t dFullBatchCount = dArgmaxActual / dProBatchSize;
    uint16_t dBlockConcurrentCount = dFullBatchCount / hwConcurrentCount;

    uint16_t dRemain = dArgmaxActual - dBlockConcurrentCount * hwConcurrentCount * dProBatchSize;

    uint16_t dRemainBatchCount = dRemain / dProBatchSize;
    uint16_t dRemainTail = dRemain - dRemainBatchCount * dProBatchSize;

    uint16_t hRemainTail = hArgmaxActual - hFullBatchCount * hProBatchSize;

    uint32_t highArgmaxStride = dhwPlaneAligned_;
    uint32_t highOutputStride = dOutputActual * hOutputActual * wOutputActual;

    uint32_t mask0 = hwConcurrentCount * hwFullBatchCount;
    uint32_t mask1 = hwConcurrentCount * hFullBatchCount;
    uint32_t mask2 = hwConcurrentCount * wFullBatchCount;
    uint32_t mask3 = hwConcurrentCount;

    uint32_t mask4 = dRemainBatchCount * hwFullBatchCount;
    uint32_t mask5 = dRemainBatchCount * hFullBatchCount;
    uint32_t mask6 = dRemainBatchCount * wFullBatchCount;
    uint32_t mask7 = dRemainBatchCount;

    uint32_t mask8 = hwFullBatchCount;
    uint32_t mask9 = hFullBatchCount;
    uint32_t mask10 = wFullBatchCount;
    uint32_t mask11 = 1;

    DivMagic divW = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount));
    DivMagic divH = PrecomputeDiv(static_cast<uint32_t>(hFullBatchCount));
    DivMagic divWH = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount * hFullBatchCount));
    DivMagic div1 = PrecomputeDiv(1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexDw;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOneDw;
        GenInitial3DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize,
                                wProBatchSize, hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned, divWH,
                                divW);
        Gen3DIndexOneFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize,
                          wArgmaxAligned, hFullBatchCount, hArgmaxActual, divH);
        GenInitial3DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize,
                                hProBatchSize, wProBatchSize, 1, hArgmaxActual, wFullBatchCount, wArgmaxAligned, divW,
                                divW);
        Gen3DIndexOneFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize,
                          wArgmaxAligned, 1, hArgmaxActual, div1);
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        DhwGenInitial2DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize,
                                   hProBatchSize, wArgmaxAligned, wFullBatchCount, divW);
        DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hProBatchSize, wArgmaxAligned);
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        uint32_t highArgmaxOffset = 0;
        uint32_t highOutputOffset = 0;
        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            for (uint16_t dIdx = 0; dIdx < dBlockConcurrentCount; dIdx++) {
                for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
                    T2 dhOffset = dProBatchIdx * hArgmaxActual * wArgmaxAligned +
                                  dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount +
                                  highArgmaxOffset;
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        T2 hOffset = hProBatchIdx * wArgmaxAligned + dhOffset;
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = wBatchIdx + hOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0,
                                                         highOutputOffset);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1,
                                                         highOutputOffset);
                        }
                    }
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                        T2 hOffset = (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned + dhOffset;
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = wBatchIdx + hOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2,
                                                         highOutputOffset);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, offset, allMaskU32);
                            DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3,
                                                         highOutputOffset);
                        }
                    }
                }
            }

            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
                T2 dhOffset = dProBatchIdx * hArgmaxActual * wArgmaxAligned +
                              (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual *
                                  wArgmaxAligned +
                              highArgmaxOffset;
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T2 hOffset = hProBatchIdx * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5,
                                                     highOutputOffset);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    T2 hOffset = (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7,
                                                     highOutputOffset);
                    }
                }
            }

            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dRemainTail; dProBatchIdx++) {
                T2 dhOffset = dProBatchIdx * hArgmaxActual * wArgmaxAligned +
                              (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize *
                                  hArgmaxActual * wArgmaxAligned +
                              highArgmaxOffset;
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T2 hOffset = hProBatchIdx * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask8,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask9,
                                                     highOutputOffset);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    T2 hOffset = (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask10,
                                                     highOutputOffset);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        DoSingleNchwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask11,
                                                     highOutputOffset);
                    }
                }
            }
            highArgmaxOffset += highArgmaxStride;
            highOutputOffset += highOutputStride;
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void
MaxPool3DGradWithArgmaxNCDHWFullLoadKernel<T1, T2, IS_CHECK_RANGE>::fullLoadMultipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
{
    int64_t wOutputActual = wOutputActual_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    int32_t highOutputPlaneActual = wOutputActual * hOutputActual * dOutputActual;
    int64_t highAxisActual = highAxisActual_;

    int64_t wArgmaxAligned = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t dArgmaxActual = dArgmaxActual_;
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint16_t dProBatchSize = curDProBatchSize_;

    uint32_t wFullBatchCount = wArgmaxActual_ / wProBatchSize;
    uint16_t wRemainTail = wArgmaxActual_ - (wProBatchSize * wFullBatchCount);
    uint32_t dFullBatchCount = dArgmaxActual / dProBatchSize;
    uint16_t dRemainTail = dArgmaxActual - (dProBatchSize * dFullBatchCount);
    uint32_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t hRemainTail = hArgmaxActual - (hProBatchSize * hFullBatchCount);

    uint16_t highConcurrentCount = V_REG_SIZE / (wFullBatchCount * hFullBatchCount * dFullBatchCount * sizeof(T2));
    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;

    int64_t depthStride = hArgmaxActual * wArgmaxAligned * dProBatchSize;
    int64_t highStride = dhwPlaneAligned_;

    uint32_t highArgmaxStride = highConcurrentCount * dhwPlaneAligned_;
    uint32_t highOutputStride = highConcurrentCount * dOutputActual * hOutputActual * wOutputActual;

    uint32_t mask0 = highConcurrentCount * dFullBatchCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask1 = highConcurrentCount * dFullBatchCount * hFullBatchCount;
    uint32_t mask2 = highConcurrentCount * dFullBatchCount * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * dFullBatchCount;
    uint32_t mask4 = highConcurrentCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask5 = highConcurrentCount * hFullBatchCount;
    uint32_t mask6 = highConcurrentCount * wFullBatchCount;
    uint32_t mask7 = highConcurrentCount;
    uint32_t mask8 = highBlockRemainTail * dFullBatchCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask9 = highBlockRemainTail * dFullBatchCount * hFullBatchCount;
    uint32_t mask10 = highBlockRemainTail * dFullBatchCount * wFullBatchCount;
    uint32_t mask11 = highBlockRemainTail * dFullBatchCount;
    uint32_t mask12 = highBlockRemainTail * hFullBatchCount * wFullBatchCount;
    uint32_t mask13 = highBlockRemainTail * hFullBatchCount;
    uint32_t mask14 = highBlockRemainTail * wFullBatchCount;
    uint32_t mask15 = highBlockRemainTail;

    uint32_t divisor_dhw = dFullBatchCount * hFullBatchCount * wFullBatchCount;
    uint32_t divisor_dh = dFullBatchCount * hFullBatchCount;
    uint32_t divisor_dw = dFullBatchCount * wFullBatchCount;
    uint32_t divisor_d = dFullBatchCount;
    uint32_t divisor_hw = hFullBatchCount * wFullBatchCount;
    uint32_t divisor_h = hFullBatchCount;
    uint32_t divisor_w = wFullBatchCount;
    uint32_t divisor_1 = 1;

    uint32_t magicHigh_dhw = 0, shiftHigh_dhw = 0;
    uint32_t magicHigh_dh = 0, shiftHigh_dh = 0;
    uint32_t magicHigh_dw = 0, shiftHigh_dw = 0;
    uint32_t magicHigh_d = 0, shiftHigh_d = 0;
    uint32_t magicHigh_hw = 0, shiftHigh_hw = 0;
    uint32_t magicHigh_h = 0, shiftHigh_h = 0;
    uint32_t magicHigh_w = 0, shiftHigh_w = 0;
    uint32_t magicHigh_1 = 0, shiftHigh_1 = 0;

    GetUintDivMagicAndShift<uint32_t>(magicHigh_dhw, shiftHigh_dhw, divisor_dhw);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_dh, shiftHigh_dh, divisor_dh);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_dw, shiftHigh_dw, divisor_dw);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_d, shiftHigh_d, divisor_d);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_hw, shiftHigh_hw, divisor_hw);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_h, shiftHigh_h, divisor_h);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_w, shiftHigh_w, divisor_w);
    GetUintDivMagicAndShift<uint32_t>(magicHigh_1, shiftHigh_1, divisor_1);

    DivMagic divW = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount));
    DivMagic divH = PrecomputeDiv(static_cast<uint32_t>(hFullBatchCount));
    DivMagic divWH = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount * hFullBatchCount));
    DivMagic divHD = PrecomputeDiv(static_cast<uint32_t>(hFullBatchCount * dFullBatchCount));
    DivMagic divDHW = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount * hFullBatchCount * dFullBatchCount));
    DivMagic divWD = PrecomputeDiv(static_cast<uint32_t>(wFullBatchCount * dFullBatchCount));
    DivMagic divD = PrecomputeDiv(static_cast<uint32_t>(dFullBatchCount));
    DivMagic div1 = PrecomputeDiv(1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
        GenInitial4DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndex, wProBatchSize, hProBatchSize,
                                wArgmaxAligned, wFullBatchCount, hFullBatchCount, dFullBatchCount, depthStride,
                                highStride, divDHW, divWH, divW);
        Gen4DIndexOneFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOne, hProBatchSize, wArgmaxAligned,
                          hFullBatchCount, dFullBatchCount, depthStride, highStride, divHD, divH);
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
        GenInitial4DIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexDW, wProBatchSize,
                                hProBatchSize, wArgmaxAligned, wFullBatchCount, 1, dFullBatchCount, depthStride,
                                highStride, divWD, divW, divW);
        Gen4DIndexOneFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOneHD, hProBatchSize, wArgmaxAligned,
                          1, dFullBatchCount, depthStride, highStride, divD, div1);
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        GenInitial3DHighIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, highStride,
                                    wProBatchSize, hProBatchSize, wArgmaxAligned, wFullBatchCount, hFullBatchCount,
                                    divWH, divW);
        Gen3DHighIndexOneFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, highStride, hProBatchSize,
                              wArgmaxAligned, hFullBatchCount, divH);
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        GenInitial2DHighIndicesFast((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, highStride,
                                    wProBatchSize, wFullBatchCount, divW);
        Gen2DHighIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, highStride);
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint32_t> magicHighReg;
        AscendC::MicroAPI::RegTensor<int32_t> indexIncReg;
        AscendC::MicroAPI::Arange(indexIncReg, 0);

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t curHighArgmaxOffset = highBlockIdx * highArgmaxStride;
            uint32_t curHighOutputOffset = highBlockIdx * highOutputStride;
            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
                T2 dhOffset = dProBatchIdx * hArgmaxActual * wArgmaxAligned + curHighArgmaxOffset;
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T2 hOffset = hProBatchIdx * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dhw);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0,
                                                     curHighOutputOffset, highOutputPlaneActual,
                                                     dFullBatchCount * hFullBatchCount * wFullBatchCount, magicHighReg,
                                                     static_cast<int16_t>(shiftHigh_dhw), indexIncReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dh);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1,
                                                     curHighOutputOffset, highOutputPlaneActual,
                                                     dFullBatchCount * hFullBatchCount, magicHighReg,
                                                     static_cast<int16_t>(shiftHigh_dh), indexIncReg);
                    }
                }
                for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                    T2 hOffset = (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dhOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dw);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2,
                                                     curHighOutputOffset, highOutputPlaneActual,
                                                     dFullBatchCount * wFullBatchCount, magicHighReg,
                                                     static_cast<int16_t>(shiftHigh_dw), indexIncReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_d);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3,
                                                     curHighOutputOffset, highOutputPlaneActual, dFullBatchCount,
                                                     magicHighReg, static_cast<int16_t>(shiftHigh_d), indexIncReg);
                    }
                }
            }
        }

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t curHighArgmaxOffset = highBlockIdx * highArgmaxStride;
            uint32_t curHighOutputOffset = highBlockIdx * highOutputStride;
            for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
                T2 dOffset = (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned +
                             curHighArgmaxOffset;
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T2 hOffset = hProBatchIdx * wArgmaxAligned + dOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_hw);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4,
                                                     curHighOutputOffset, highOutputPlaneActual,
                                                     hFullBatchCount * wFullBatchCount, magicHighReg,
                                                     static_cast<int16_t>(shiftHigh_hw), indexIncReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_h);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5,
                                                     curHighOutputOffset, highOutputPlaneActual, hFullBatchCount,
                                                     magicHighReg, static_cast<int16_t>(shiftHigh_h), indexIncReg);
                    }
                }
                for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                    T2 hOffset = (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dOffset;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = wBatchIdx + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_w);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6,
                                                     curHighOutputOffset, highOutputPlaneActual, wFullBatchCount,
                                                     magicHighReg, static_cast<int16_t>(shiftHigh_w), indexIncReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_1);
                        DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7,
                                                     curHighOutputOffset, highOutputPlaneActual, 1, magicHighReg,
                                                     static_cast<int16_t>(shiftHigh_1), indexIncReg);
                    }
                }
            }
        }

        uint32_t highArgmaxOffset = highBlockConcurrentCount * highArgmaxStride;
        uint32_t highOutputOffset = highBlockConcurrentCount * highOutputStride;
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            T2 dhOffset = dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                T2 hOffset = hProBatchIdx * wArgmaxAligned + dhOffset;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = wBatchIdx + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dhw);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask8, highOutputOffset,
                                                 highOutputPlaneActual,
                                                 dFullBatchCount * hFullBatchCount * wFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_dhw), indexIncReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dh);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask9, highOutputOffset,
                                                 highOutputPlaneActual, dFullBatchCount * hFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_dh), indexIncReg);
                }
            }
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                T2 hOffset = (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dhOffset;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = wBatchIdx + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_dw);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask10,
                                                 highOutputOffset, highOutputPlaneActual,
                                                 dFullBatchCount * wFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_dw), indexIncReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_d);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask11,
                                                 highOutputOffset, highOutputPlaneActual, dFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_d), indexIncReg);
                }
            }
        }

        for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
            T2 dOffset = (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned +
                         highArgmaxOffset;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                T2 hOffset = hProBatchIdx * wArgmaxAligned + dOffset;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = wBatchIdx + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_hw);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask12,
                                                 highOutputOffset, highOutputPlaneActual,
                                                 hFullBatchCount * wFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_hw), indexIncReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_h);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask13,
                                                 highOutputOffset, highOutputPlaneActual, hFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_h), indexIncReg);
                }
            }
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                T2 hOffset = (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dOffset;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = wBatchIdx + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_w);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask14,
                                                 highOutputOffset, highOutputPlaneActual, wFullBatchCount, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_w), indexIncReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + hOffset;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(magicHighReg, magicHigh_1);
                    DoMulNCNcdhwFullLoad<T1, T2>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask15,
                                                 highOutputOffset, highOutputPlaneActual, 1, magicHighReg,
                                                 static_cast<int16_t>(shiftHigh_1), indexIncReg);
                }
            }
        }
    }
}

} // namespace MaxPool3DGradWithArgmaxNCDHWNameSpace
#endif // MAX_POOL_GRAD_WITH_ARGMAX_SIMD_IMPL_H_
