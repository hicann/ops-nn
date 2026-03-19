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
 * \file max_pool3d_grad_small_kernel.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "max_pool3d_grad_struct.h"

namespace MaxPool3DSmallKernelNameSpace {
using namespace AscendC;
using namespace Pool3DGradNameSpace;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t RATIO = 2;

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
class Pool3DGradSmallKernel {
public:
    __aicore__ inline Pool3DGradSmallKernel(TPipe& pipeIn, const Pool3DGradNCDHWTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y);
    __aicore__ inline void CopyOut();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void Process();
    __aicore__ inline void ConvertIndexWithoutPadAlign(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, uint32_t hInputActualPad, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
        TYPE_ARGMAX hIndexBase, TYPE_ARGMAX hInput, TYPE_ARGMAX dIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset);
    __aicore__ inline void ProcessW(
        __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset, uint16_t hInputActualPad,
        MicroAPI::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel,
        uint16_t repeatElem, MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t dDilation, uint32_t hDilation,
        uint32_t wDilation);
    __aicore__ inline void ConvertIndexWithoutPadAlignNc(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t hInputActualPad, TYPE_ARGMAX left, TYPE_ARGMAX wInput,
        TYPE_ARGMAX hIndexBase, TYPE_ARGMAX hInput, TYPE_ARGMAX dIndexBase, MicroAPI::RegTensor<TYPE_ARGMAX>& dstReg, int32_t ncInputOffset,
        int32_t ncOutputCount, int32_t inputNcSize);
    __aicore__ inline void MultiNcGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void MultiDepGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void MultiRowGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void SingleRowGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void DupBufferNegInf(
        __local_mem__ TYPE_ORIG_X* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail);
    __aicore__ inline void CopyToCalcBuffer(
        __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr, uint16_t batch, uint16_t deps,
        uint16_t rows, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride,
        uint32_t srcDepStride, uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstDepStride,
        uint32_t dstRowStride, uint32_t dstDepOffset, uint32_t dstRowOffset, uint32_t dstColOffset);

    __aicore__ inline void DupAndCopyToCalcBuffer(
        __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr);

    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ParseTilingData(const Pool3DGradNCDHWTilingData& tilingData);
    __aicore__ inline void singleLineProcessVF(__local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr,
                                               __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void multipleLineProcessVF2(__local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr,
                                                  __local_mem__ TYPE_ARGMAX* argmaxAddr, __local_mem__ uint32_t* helpAddr);
    __aicore__ inline void multipleLineHwProcessVF(__local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr,
                                               __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void multipleLineDhwProcessVF(__local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr,
                                                  __local_mem__ TYPE_ARGMAX* argmaxAddr);
    __aicore__ inline void ProcessNoArgmaxBlock();

    TPipe& pipe_;
    const Pool3DGradNCDHWTilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;
    TBuf<TPosition::VECCALC> argmaxBuff_;

    GlobalTensor<TYPE_ORIG_X> gradGm_;
    GlobalTensor<TYPE_ORIG_X> yGm_;
    GlobalTensor<TYPE_ARGMAX> argmaxGm_;
    GlobalTensor<TYPE_ORIG_X> xGm_;
    
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

    int64_t dInputActualPad_ = 0;
    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;
    int64_t wInputActualAlignedPad_ = 0;
    int64_t frontOffsetToInputFront_ = 0;
    int64_t backOffsetToInputBack_ = 0;
    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;

    int64_t highInputOffset_ = 0;
    int64_t forwarddInputOffset_ = 0;
    int64_t forwardhInputOffset_ = 0;
    int64_t forwardwInputOffset_ = 0;

    int64_t dInputActualNoPad_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
    int64_t wOutputActualAligned_ = 0;

    int64_t forwardHighAxisIndex_ = 0;
    int64_t forwardhighAxisActual_ = 0;
    int64_t forwardDAxisIndex_ = 0;
    int64_t dOutputReal_ = 0;
    int64_t forwardHAxisIndex_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t forwardWAxisIndex_ = 0;
    int64_t wOutputReal_ = 0;

    int64_t dArgmaxActualStart = 0;
    int64_t dArgmaxActualEnd = 0;
    int64_t hArgmaxActualStart = 0;
    int64_t hArgmaxActualEnd = 0;
    int64_t wArgmaxActualStart = 0;
    int64_t wArgmaxActualEnd = 0;   

    bool IS_PAD = false;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(TYPE_ORIG_X) >= BLOCK_SIZE / sizeof(TYPE_ARGMAX) ? BLOCK_SIZE / sizeof(TYPE_ORIG_X) : BLOCK_SIZE / sizeof(TYPE_ARGMAX);
    constexpr static uint16_t vlT2_ = platform::GetVRegSize() / sizeof(TYPE_ARGMAX);
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(TYPE_ORIG_X);
};

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Init(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y)
{
    ParseTilingData(tilingData_);
    blockIdx_ = GetBlockIdx();
    argmaxPlaneSize_ = dArgmax_ * hArgmax_ * wArgmax_;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    xGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)orig_x);
    gradGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)grads);
    yGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)y);

    pipe_.InitBuffer(inputQue_, BUFFER_NUM, tilingData_.inputBufferSize);
    if (IS_PAD) {
        pipe_.InitBuffer(inputCalcBuff_, tilingData_.inputBufferSize);
    }
    pipe_.InitBuffer(argmaxBuff_, tilingData_.argmaxBufferSize);
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputBufferSize_);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, gradBufferSize_);
    pipe_.InitBuffer(helpBuf_, HELP_BUFFER);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Compute()
{
    LocalTensor<TYPE_ORIG_X> inputLocal = inputQue_.DeQue<TYPE_ORIG_X>();
    LocalTensor<TYPE_ORIG_X> caclBuffLocal;
    __local_mem__ TYPE_ORIG_X* inputBuffAddr;
    __local_mem__ TYPE_ORIG_X* inputQueAddr = (__local_mem__ TYPE_ORIG_X*)inputLocal.GetPhyAddr();
    __local_mem__ TYPE_ORIG_X* computeAddr = inputQueAddr;
    if (IS_PAD) {
        caclBuffLocal = inputCalcBuff_.Get<TYPE_ORIG_X>();
        inputBuffAddr = (__local_mem__ TYPE_ORIG_X*)caclBuffLocal.GetPhyAddr();
        DupAndCopyToCalcBuffer(inputBuffAddr, inputQueAddr);
        computeAddr = inputBuffAddr;
    }
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    LocalTensor<TYPE_ORIG_X> gradLocal = gradQue_.DeQue<TYPE_ORIG_X>();
    LocalTensor<TYPE_ARGMAX> argmaxLocal = argmaxBuff_.Get<TYPE_ARGMAX>();
    Duplicate(argmaxLocal, TYPE_ARGMAX(0), tilingData_.argmaxBufferSize / sizeof(TYPE_ARGMAX));
    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ TYPE_ORIG_X* gradAddr = (__local_mem__ TYPE_ORIG_X*)gradLocal.GetPhyAddr();
    __local_mem__ TYPE_ARGMAX* argmaxAddr = (__local_mem__ TYPE_ARGMAX*)argmaxLocal.GetPhyAddr();
    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    uint32_t dConcurrentCount = dArgmaxActual_ / curDProBatchSize_;

    if (wOutputActual_ * RATIO > vlT2_) {
        SingleRowGather(computeAddr,argmaxAddr);
    } else if (hOutputActual_ * wOutputActual_ * RATIO > vlT2_) {
        MultiRowGather(computeAddr,argmaxAddr);
    } else if( dOutputActual_ * hOutputActual_ * wOutputActual_ * RATIO > vlT2_){
        MultiDepGather(computeAddr,argmaxAddr);
    }else {
        MultiNcGather(computeAddr, argmaxAddr);
    } 

    if (wConcurrentCount * DOUBLE * sizeof(TYPE_ARGMAX) > V_REG_SIZE) {
        singleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(TYPE_ARGMAX) > V_REG_SIZE) {
        multipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if(wConcurrentCount * hConcurrentCount * dConcurrentCount * DOUBLE * sizeof(TYPE_ARGMAX) > V_REG_SIZE) {
        multipleLineDhwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else {   
        LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
        __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
        multipleLineProcessVF2(yAddr, gradAddr, argmaxAddr, helpAddr);
    }

    inputQue_.FreeTensor(inputLocal);
    if constexpr (std::negation<std::is_same<TYPE_ORIG_X, float>>::value) {
        Cast(yLocal.ReinterpretCast<TYPE_ORIG_X>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ParseTilingData(
    const Pool3DGradNCDHWTilingData& tilingData)
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
    IS_PAD = tilingData_.padD != 0 || tilingData_.padH != 0 || tilingData_.padW != 0;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{   
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;  
    highAxisIndex_ = baseBlockIdx / (dOutputOuter_ * hOutputOuter_ * wOutputOuter_); 
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;
    int64_t tempTail = baseBlockIdx % (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);  
    dAxisIndex_ = tempTail / (hOutputOuter_ * wOutputOuter_); 
    dOutputActual_ = dAxisIndex_ == (dOutputOuter_ - 1) ? dOutputTail_ : dOutputInner_; 
    int64_t tempTail2 = tempTail % (hOutputOuter_ * wOutputOuter_);  
    hAxisIndex_ = tempTail2 / wOutputOuter_;  
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;
    wAxisIndex_ = tempTail2 % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;

    wOutputAligned_ =
        (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;
    dArgmaxActualStart = PStart(dAxisIndex_ * dOutputInner_, padD_, kernelD_, dilationD_, strideD_);            //当处理的大窗口大小
    dArgmaxActualEnd = PEnd(dAxisIndex_ * dOutputInner_ + dOutputActual_ - 1, padD_, strideD_, dArgmax_);
    hArgmaxActualStart = PStart(hAxisIndex_ * hOutputInner_, padH_, kernelH_, dilationH_, strideH_);
    hArgmaxActualEnd = PEnd(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, padH_, strideH_, hArgmax_);
    wArgmaxActualStart = PStart(wAxisIndex_ * wOutputInner_, padW_, kernelW_, dilationW_, strideW_);
    wArgmaxActualEnd = PEnd(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, padW_, strideW_, wArgmax_);
    wArgmaxActual_ = wArgmaxActualEnd - wArgmaxActualStart;
    wArgmaxAligned_ =
        (wArgmaxActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;
    hArgmaxActual_ = hArgmaxActualEnd - hArgmaxActualStart; 
    dArgmaxActual_ = dArgmaxActualEnd - dArgmaxActualStart; 
    
    curDProBatchSize_ = dProBatchSize_ > dArgmaxActual_ ? dArgmaxActual_ : dProBatchSize_;
    curHProBatchSize_ = hProBatchSize_ > hArgmaxActual_ ? hArgmaxActual_ : hProBatchSize_;
    curWProBatchSize_ = wProBatchSize_ > wArgmaxActual_ ? wArgmaxActual_ : wProBatchSize_;
    highAxisArgmaxOffset_ = highAxisIndex_ * highAxisInner_ * argmaxPlaneSize_;
    dAxisArgmaxOffset_ = dArgmaxActualStart * hArgmax_ * wArgmax_;
    hAxisArgmaxOffset_ = hArgmaxActualStart * wArgmax_;
    wAxisArgmaxOffset_ = wArgmaxActualStart;
    
    int64_t gradDOuter = dArgmax_ / dArgmaxActual_;
    int64_t gradHOuter = hArgmax_ / hArgmaxActual_;
    int64_t gradWOuter = wArgmax_ / wArgmaxActual_;   
    int64_t d_h_w_outer = gradDOuter * gradHOuter * gradWOuter;
    int64_t h_w_outer = gradHOuter * gradWOuter;
    int64_t w_outer = gradWOuter;

    forwardHighAxisIndex_ = baseBlockIdx / d_h_w_outer;
    forwardhighAxisActual_ =
        forwardHighAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail : tilingData_.highAxisInner;

    int64_t base_mod_dhw = baseBlockIdx - d_h_w_outer * (baseBlockIdx / d_h_w_outer);
    forwardDAxisIndex_ = base_mod_dhw / h_w_outer;
    dOutputReal_ = dArgmaxActual_;

    int64_t forwardtempTail = baseBlockIdx - h_w_outer * (baseBlockIdx / h_w_outer);
    forwardHAxisIndex_ = forwardtempTail / w_outer;
    hOutputReal_ = hArgmaxActual_;
    
    forwardWAxisIndex_ = forwardtempTail - w_outer * (forwardtempTail / w_outer);
    wOutputReal_ = wArgmaxActual_;

    wOutputActualAligned_ = CeilDivision(wOutputReal_, MAX_DATA_NUM_IN_ONE_BLOCK) * MAX_DATA_NUM_IN_ONE_BLOCK;
    dInputActualPad_ =
        (dOutputReal_ - 1) * tilingData_.dStride + (tilingData_.dKernel - 1) * tilingData_.dilationD + 1;
    hInputActualPad_ =
        (hOutputReal_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.dilationH + 1;
    wInputActualPad_ =
        (wOutputReal_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.dilationW + 1;

    wInputActualAlignedPad_ = CeilDivision(wInputActualPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) * (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
    int64_t inputPlaneSize =  tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput;
    highInputOffset_ = highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    forwarddInputOffset_ = dArgmaxActualStart * tilingData_.dStride * tilingData_.hOutput * tilingData_.wOutput;
    forwardhInputOffset_ = hArgmaxActualStart * tilingData_.hStride * tilingData_.wOutput;
    forwardwInputOffset_ = wArgmaxActualStart * tilingData_.wStride;

    if (IS_PAD) {
        int64_t tRelBoundDistance =
            hArgmaxActualStart * tilingData_.hStride - tilingData_.padH;
        int64_t dRelBoundDistance = hArgmaxActualStart * tilingData_.hStride +
                                       (hOutputReal_ - 1) * tilingData_.hStride + tilingData_.hKernel -
                                       tilingData_.hOutput - tilingData_.padH;
        int64_t lRelBoundDistance =
            wArgmaxActualStart * tilingData_.wStride - tilingData_.padW;
        int64_t rRelBoundDistance = wArgmaxActualStart * tilingData_.wStride +
                                       (wOutputReal_ - 1) * tilingData_.wStride + tilingData_.wKernel -
                                       tilingData_.wOutput - tilingData_.padW;
        int64_t frontRelBoundDistance =
            dArgmaxActualStart * tilingData_.dStride - tilingData_.padD;
        int64_t backRelBoundDistance = dArgmaxActualStart * tilingData_.dStride +
                                       (dOutputReal_ - 1) * tilingData_.dStride + tilingData_.dKernel -
                                       tilingData_.dOutput - tilingData_.padD;
        leftOffsetToInputLeft_ = lRelBoundDistance >= 0 ? 0 : -lRelBoundDistance;
        rightOffsetToInputRight_ = rRelBoundDistance >= 0 ? rRelBoundDistance : 0;
        topOffsetToInputTop_ = tRelBoundDistance >= 0 ? 0 : -tRelBoundDistance;
        downOffsetToInputDown_ = dRelBoundDistance >= 0 ? dRelBoundDistance : 0;
        frontOffsetToInputFront_ = frontRelBoundDistance >= 0 ? 0 : -frontRelBoundDistance;
        backOffsetToInputBack_ = backRelBoundDistance >= 0 ? backRelBoundDistance : 0;
        dInputActualNoPad_ = dInputActualPad_ - frontOffsetToInputFront_ - backOffsetToInputBack_;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
        forwarddInputOffset_ = frontOffsetToInputFront_ == 0 ? forwarddInputOffset_ - tilingData_.padD * tilingData_.hOutput * tilingData_.wOutput : 0;
        forwardhInputOffset_ = topOffsetToInputTop_ == 0 ? forwardhInputOffset_ - tilingData_.padH * tilingData_.wOutput : 0;
        forwardwInputOffset_ = leftOffsetToInputLeft_ == 0 ? forwardwInputOffset_ - tilingData_.padW : 0;
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyIn()
{
    LocalTensor<TYPE_ORIG_X> gradLocal = gradQue_.AllocTensor<TYPE_ORIG_X>();
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.AllocTensor<TYPE_ORIG_X>();
    int64_t xGmOffset = highInputOffset_ + forwarddInputOffset_ + forwardhInputOffset_ + forwardwInputOffset_;
    int64_t planeHW = hArgmax_ * wArgmax_;
    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + dAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_; 
    DataCopyPadExtParams<TYPE_ORIG_X> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = dArgmaxActual_;
    loopModeParamsT1.loop2Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = planeHW * sizeof(TYPE_ORIG_X); 
    loopModeParamsT1.loop2SrcStride = argmaxPlaneSize_ * sizeof(TYPE_ORIG_X);     
    loopModeParamsT1.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(TYPE_ORIG_X);
    loopModeParamsT1.loop2DstStride = dArgmaxActual_ * hArgmaxActual_ * wArgmaxAligned_ * sizeof(TYPE_ORIG_X);

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(hArgmaxActual_), 
        static_cast<uint32_t>(wArgmaxActual_ * sizeof(TYPE_ORIG_X)), 
        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(TYPE_ORIG_X)), 
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);

    LoopModeParams loopModeParamsT2;
    int64_t wInputActualAlignedNoPadTmp;
    if (IS_PAD){
        int64_t wInputActualAlignedNoPad = CeilDivision(wInputActualNoPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) * (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
        wInputActualAlignedNoPadTmp = wInputActualAlignedNoPad;
        loopModeParamsT2.loop1Size = highAxisActual_;
        loopModeParamsT2.loop2Size = dInputActualNoPad_;
        loopModeParamsT2.loop1SrcStride = tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput * sizeof(TYPE_ORIG_X); 
        loopModeParamsT2.loop2SrcStride = tilingData_.hOutput * tilingData_.wOutput * sizeof(TYPE_ORIG_X);
        loopModeParamsT2.loop1DstStride = dInputActualNoPad_ * hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(TYPE_ORIG_X);
        loopModeParamsT2.loop2DstStride = hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(TYPE_ORIG_X);
    }else {
        loopModeParamsT2.loop1Size = highAxisActual_;
        loopModeParamsT2.loop2Size = dInputActualPad_;
        loopModeParamsT2.loop1SrcStride = tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput * sizeof(TYPE_ORIG_X);
        loopModeParamsT2.loop2SrcStride = tilingData_.hOutput * tilingData_.wOutput * sizeof(TYPE_ORIG_X);
        loopModeParamsT2.loop1DstStride = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_ * sizeof(TYPE_ORIG_X);
        loopModeParamsT2.loop2DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(TYPE_ORIG_X);
    }

    SetLoopModePara(loopModeParamsT2, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<TYPE_ORIG_X> paramsT2 = {false, 0, 0, 0};
    DataCopyExtParams copyOutParamT2;
    if (IS_PAD) { 
        copyOutParamT2.blockCount = static_cast<uint16_t>(hInputActualNoPad_); 
        copyOutParamT2.blockLen = static_cast<uint32_t>(wInputActualNoPad_ * sizeof(TYPE_ORIG_X));
        copyOutParamT2.srcStride = static_cast<uint32_t>((tilingData_.wOutput - wInputActualNoPad_) * sizeof(TYPE_ORIG_X));
        copyOutParamT2.dstStride = 0;
        copyOutParamT2.rsv = 0;
    } else {
        copyOutParamT2.blockCount = static_cast<uint16_t>(hInputActualPad_); 
        copyOutParamT2.blockLen = static_cast<uint32_t>(wInputActualPad_ * sizeof(TYPE_ORIG_X)); 
        copyOutParamT2.srcStride = static_cast<uint32_t>((tilingData_.wOutput - wInputActualPad_) * sizeof(TYPE_ORIG_X));
        copyOutParamT2.dstStride = 0;
        copyOutParamT2.rsv = 0;
    }
    DataCopyPad(xLocal, xGm_[xGmOffset], copyOutParamT2, paramsT2);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<TYPE_ORIG_X> yLocal = outputQue_.DeQue<TYPE_ORIG_X>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t outputPlaneDHW = dOutput_ * outputPlaneSize; 
    int64_t ncBase = highAxisIndex_ * highAxisInner_ * outputPlaneDHW; 
    int64_t dBase = dAxisIndex_ * dOutputInner_ * outputPlaneSize;
    int64_t hBase = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wBase = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = ncBase + dBase + hBase + wBase;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = dOutputActual_;
    loopModeParamsT1.loop2Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(TYPE_ORIG_X);
    loopModeParamsT1.loop2SrcStride = dOutputActual_ * hOutputActual_ * wOutputAligned_ * sizeof(TYPE_ORIG_X);
    loopModeParamsT1.loop1DstStride = outputPlaneSize * sizeof(TYPE_ORIG_X);
    loopModeParamsT1.loop2DstStride = outputPlaneDHW * sizeof(TYPE_ORIG_X); 

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(TYPE_ORIG_X)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(TYPE_ORIG_X)),
                                        static_cast<uint32_t>(0)};
    
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0 || dArgmaxActual_ <= 0) {
            ProcessNoArgmaxBlock();
            continue;
        }
        CopyIn();
        Compute();
        CopyOut();
    }
}
}
#endif // MAX_POOL3D_GRAD_SMALL_KERNEL_H
