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
 * \file max_pool3d_with_argmax_v2_gather_kernel.h
 * \brief
 */
#ifndef MAX_POOL_3D_WITH_ARGMAX_V2_GATHER_KERNEL_H_
#define MAX_POOL_3D_WITH_ARGMAX_V2_GATHER_KERNEL_H_

#include "max_pool3d_with_argmax_v2_base.h"
#include "max_pool3d_with_argmax_v2_tiling_struct.h"

namespace MaxPool3DWithArgmaxV2GatherNameSpace
{
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t HELPER_BUFFER_SIZE = 1024;
constexpr int64_t THREE_DIM = 3;
constexpr int64_t RATIO = 2;
constexpr uint16_t B32 = 4;


template <typename T1, typename T2, const uint32_t IS_PAD = 0>
class MaxPool3DWithArgmaxV2GatherKernel
{
public:
    __aicore__ inline MaxPool3DWithArgmaxV2GatherKernel(TPipe& pipeIn,
                                                      const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessEachLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void DupBufferNegInf(__local_mem__ T1* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail);
    __aicore__ inline void CopyToCalcBuffer(__local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch,uint16_t deps ,uint16_t rows, uint16_t loopCols,
                                            uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride,uint32_t srcRowStride, 
                                            uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride, uint32_t dstDepOffset,
                                            uint32_t dstRowOffset, uint32_t dstColOffset);
    __aicore__ inline void DupAndCopyToCalcBuffer(__local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr);
    __aicore__ inline void ConvertIndexWithoutPadAlign(MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, uint32_t hInputActualPad, 
                                                        T2 left, T2 wInput, T2 hIndexBase, T2 hInput,T2 dIndexBase,
                                                        MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset);
    __aicore__ inline void ConvertIndexWithoutPadAlignNc(MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,int32_t hInputActualPad, T2 left,
                                                        T2 wInput, T2 hIndexBase,T2 hInput, T2 dIndexBase, MicroAPI::RegTensor<T2>& dstReg, 
                                                        int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize);
    __aicore__ inline void ProcessW(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, int32_t hOffset, uint16_t wStrideOffset, uint16_t hInputActualPad,
                                    MicroAPI::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
                                    int32_t outputOffset, MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t dDilation ,uint32_t hDilation, uint32_t wDilation);
                                    
    __aicore__ inline void SingleRowGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                           __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiRowGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                          __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiDepGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                          __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiNcGather(__local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr,
                                         __local_mem__ T2* argmaxAddr);

private:
    TPipe& pipe_;
    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxValueQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;
    int64_t highAxisActual_ = 0;
    int64_t dOutputReal_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t wOutputReal_ = 0;
    int64_t curCoreProcessNum_ = 0;
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

    int64_t highAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t highInputOffset_ = 0;
    int64_t dInputOffset_ = 0;
    int64_t hInputOffset_ = 0;
    int64_t wInputOffset_ = 0;

    int64_t dInputActualNoPad_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
    int64_t wOutputActualAligned_ = 0;

    constexpr static int32_t blockSize_ = platform::GetUbBlockSize();
    
    constexpr static int64_t maxDataNumOneBlock_ =
        blockSize_ / sizeof(T1) >= blockSize_ / sizeof(T2) ? blockSize_ / sizeof(T1) : blockSize_ / sizeof(T2);
    constexpr static uint16_t vlT2_ = platform::GetVRegSize() / sizeof(T2);
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(T1);
};

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    curCoreProcessNum_ =
        (blockIdx_ + 1 == tilingData_.usedCoreNum) ? tilingData_.tailCoreProcessNum : tilingData_.normalCoreProcessNum;
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);
    
    pipe_.InitBuffer(inputQue_, BUFFER_NUM, tilingData_.inputBufferSize);
    if constexpr (IS_PAD == 1) {
        pipe_.InitBuffer(inputCalcBuff_, tilingData_.inputBufferSize);
    }
    pipe_.InitBuffer(maxValueQue_, BUFFER_NUM, tilingData_.maxValueBufferSize);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, tilingData_.argmaxBufferSize);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessEachLoop();
    }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::ScalarCompute(int64_t loopNum)
{
    int64_t d_h_w_outer = tilingData_.dOutputOuter * tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t h_w_outer = tilingData_.hOutputOuter * tilingData_.wOutputOuter;
    int64_t w_outer = tilingData_.wOutputOuter;

    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;

    highAxisIndex_ = baseBlockIdx / d_h_w_outer;
    highAxisActual_ =
        highAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail : tilingData_.highAxisInner;

    int64_t base_mod_dhw = baseBlockIdx - d_h_w_outer * (baseBlockIdx / d_h_w_outer);
    dAxisIndex_ = base_mod_dhw / h_w_outer;
    dOutputReal_ = dAxisIndex_ == (tilingData_.dOutputOuter - 1) ? tilingData_.dOutputTail : tilingData_.dOutputInner;

    int64_t tempTail = baseBlockIdx - h_w_outer * (baseBlockIdx / h_w_outer);
    hAxisIndex_ = tempTail / w_outer;
    hOutputReal_ = hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner;

    
    wAxisIndex_ = tempTail - w_outer * (tempTail / w_outer);
    wOutputReal_ = wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner;

    wOutputActualAligned_ = CeilDivision(wOutputReal_, maxDataNumOneBlock_) * maxDataNumOneBlock_;
    dInputActualPad_ =
        (dOutputReal_ - 1) * tilingData_.dStride + (tilingData_.dKernel - 1) * tilingData_.dDilation + 1;
    hInputActualPad_ =
        (hOutputReal_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.hDilation + 1;
    wInputActualPad_ =
        (wOutputReal_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.wDilation + 1;

    wInputActualAlignedPad_ = CeilDivision(wInputActualPad_, blockSize_ / sizeof(T1)) * (blockSize_ / sizeof(T1));
    int64_t inputPlaneSize =  tilingData_.dInput * tilingData_.hInput * tilingData_.wInput;
    highInputOffset_ = highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    dInputOffset_ = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride * tilingData_.hInput * tilingData_.wInput;
    hInputOffset_ = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride * tilingData_.wInput;
    wInputOffset_ = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride;

    if constexpr (IS_PAD == 1) {
        int64_t tRelBoundDistance =
            hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;

        int64_t dRelBoundDistance = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride +
                                       (hOutputReal_ - 1) * tilingData_.hStride + tilingData_.hKernel -
                                       tilingData_.hInput - tilingData_.padTop;
        int64_t lRelBoundDistance =
            wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
        int64_t rRelBoundDistance = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride +
                                       (wOutputReal_ - 1) * tilingData_.wStride + tilingData_.wKernel -
                                       tilingData_.wInput - tilingData_.padLeft;
        int64_t frontRelBoundDistance =
            dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
        int64_t backRelBoundDistance = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride +
                                       (dOutputReal_ - 1) * tilingData_.dStride + tilingData_.dKernel -
                                       tilingData_.dInput - tilingData_.padFront;
        leftOffsetToInputLeft_ = lRelBoundDistance >= 0 ? 0 : -lRelBoundDistance;
        rightOffsetToInputRight_ = rRelBoundDistance >= 0 ? rRelBoundDistance : 0;
        topOffsetToInputTop_ = tRelBoundDistance >= 0 ? 0 : -tRelBoundDistance;
        downOffsetToInputDown_ = dRelBoundDistance >= 0 ? dRelBoundDistance : 0;

        frontOffsetToInputFront_ = frontRelBoundDistance >= 0 ? 0 : -frontRelBoundDistance;
        backOffsetToInputBack_ = backRelBoundDistance >= 0 ? backRelBoundDistance : 0;

        dInputActualNoPad_ = dInputActualPad_ - frontOffsetToInputFront_ - backOffsetToInputBack_;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;

        dInputOffset_ = frontOffsetToInputFront_ == 0 ? dInputOffset_ - tilingData_.padFront * tilingData_.hInput * tilingData_.wInput : 0;
        hInputOffset_ = topOffsetToInputTop_ == 0 ? hInputOffset_ - tilingData_.padTop * tilingData_.wInput : 0;
        wInputOffset_ = leftOffsetToInputLeft_ == 0 ? wInputOffset_ - tilingData_.padLeft : 0;
    }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::ProcessEachLoop()
{
    CopyIn();
    Compute();
    CopyOut();
}


// kernel 
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::CopyIn()
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    int64_t xGmOffset = highInputOffset_ + dInputOffset_ + hInputOffset_ + wInputOffset_;
    
    LoopModeParams loopModeParamsT1;
    // 
    if constexpr(IS_PAD == 1){
        int64_t wInputActualAlignedNoPad = CeilDivision(wInputActualNoPad_, blockSize_ / sizeof(T1)) * (blockSize_ / sizeof(T1));
        loopModeParamsT1.loop1Size = highAxisActual_;
        loopModeParamsT1.loop2Size = dInputActualNoPad_;
        loopModeParamsT1.loop1SrcStride = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput * sizeof(T1); 
        loopModeParamsT1.loop2SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop1DstStride = dInputActualNoPad_ * hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(T1);
        loopModeParamsT1.loop2DstStride = hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(T1);
    }else {
        loopModeParamsT1.loop1Size = highAxisActual_;
        loopModeParamsT1.loop2Size = dInputActualPad_;
        loopModeParamsT1.loop1SrcStride = tilingData_.dInput * tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop2SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
        loopModeParamsT1.loop1DstStride = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_ * sizeof(T1);
        loopModeParamsT1.loop2DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(T1);
    }

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    DataCopyExtParams copyOutParamT1;
    if constexpr (IS_PAD == 1) { 
        copyOutParamT1.blockCount = static_cast<uint16_t>(hInputActualNoPad_); 
        copyOutParamT1.blockLen = static_cast<uint32_t>(wInputActualNoPad_ * sizeof(T1));
        copyOutParamT1.srcStride = static_cast<uint32_t>((tilingData_.wInput - wInputActualNoPad_) * sizeof(T1));
        copyOutParamT1.dstStride = 0;
        copyOutParamT1.rsv = 0;
    } else {
        copyOutParamT1.blockCount = static_cast<uint16_t>(hInputActualPad_); 
        copyOutParamT1.blockLen = static_cast<uint32_t>(wInputActualPad_ * sizeof(T1)); 
        copyOutParamT1.srcStride = static_cast<uint32_t>((tilingData_.wInput - wInputActualPad_) * sizeof(T1));
        copyOutParamT1.dstStride = 0;
        copyOutParamT1.rsv = 0;
    }
    DataCopyPad(xLocal, xGm_[xGmOffset], copyOutParamT1, paramsT1);
    inputQue_.EnQue(xLocal);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::DupBufferNegInf(__local_mem__ T1* dstAddr,
                                                                                        uint32_t repeatElm,
                                                                                        uint16_t loop, uint32_t tail)
{
    MicroAPI::RegTensor<T1> v0;
    SetNegInfReg<T1>(v0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    uint32_t maskCount = tail;
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = MicroAPI::UpdateMask<T1>(maskCount);
    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::CopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch,uint16_t deps ,uint16_t rows, uint16_t loopCols,
    uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride,uint32_t srcRowStride, 
    uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride, uint32_t dstDepOffset, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    MicroAPI::RegTensor<T1> v0;
    MicroAPI::UnalignReg u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t t = 0; t < deps; t++) {
            for (uint16_t j = 0; j < rows; j++) {
                __local_mem__ T1* curSrcAddr = srcAddr + i * srcBatchStride + t * srcDepStride + j * srcRowStride;
                __local_mem__ T1* curDstAddr =
                    dstAddr + i * dstBatchStride + (t + dstDepOffset) * dstDepStride +(j + dstRowOffset) * dstRowStride + dstColOffset;
                for (uint16_t k = 0; k < loopCols; k++) {
                    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
            }
        }
    }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::DupAndCopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr)
{
    uint16_t loopCols = wInputActualNoPad_ / vlT1_;
    uint16_t tailCols = wInputActualNoPad_ - loopCols * vlT1_;
    uint32_t wInputActualNoPadAlign =
        CeilDivision(wInputActualNoPad_, blockSize_ / sizeof(T1)) * blockSize_ / sizeof(T1);
    uint32_t totalInputPad = tilingData_.highAxisInner * dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t loopDup = totalInputPad / vlT1_;
    uint32_t tailDup = totalInputPad - loopDup * vlT1_;
    uint32_t dstDepOffset = frontOffsetToInputFront_;
    uint32_t dstRowOffset = topOffsetToInputTop_;
    uint32_t dstColOffset = leftOffsetToInputLeft_;
    uint32_t srcDepStride =  hInputActualNoPad_ * wInputActualNoPadAlign;
    uint32_t srcBatchStride = dInputActualNoPad_ * hInputActualNoPad_ * wInputActualNoPadAlign;
    uint32_t dstDepStride =  hInputActualPad_ * wInputActualAlignedPad_;
    uint32_t dstBatchStride = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    __VEC_SCOPE__
    {
        DupBufferNegInf(dstAddr, vlT1_, loopDup, tailDup);
        CopyToCalcBuffer(dstAddr, srcAddr, highAxisActual_, dInputActualNoPad_,hInputActualNoPad_, loopCols, 
                        tailCols, vlT1_, srcBatchStride, srcDepStride, wInputActualNoPadAlign, 
                        dstBatchStride, dstDepStride, wInputActualAlignedPad_,dstDepOffset,dstRowOffset, dstColOffset);
    }
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::Compute()
{
    LocalTensor<T1> inputLocal = inputQue_.DeQue<T1>();
    LocalTensor<T1> caclBuffLocal;
    __local_mem__ T1* inputBuffAddr;
    __local_mem__ T1* inputQueAddr = (__local_mem__ T1*)inputLocal.GetPhyAddr();
    __local_mem__ T1* computeAddr = inputQueAddr;
    if constexpr (IS_PAD == 1) {
        caclBuffLocal = inputCalcBuff_.Get<T1>();
        inputBuffAddr = (__local_mem__ T1*)caclBuffLocal.GetPhyAddr();
        DupAndCopyToCalcBuffer(inputBuffAddr, inputQueAddr);
        computeAddr = inputBuffAddr;
    }
    LocalTensor<T1> maxValueLocal = maxValueQue_.AllocTensor<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();
    __local_mem__ T1* maxValueAddr = (__local_mem__ T1*)maxValueLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();
    

    if (wOutputReal_ * RATIO > vlT2_) {
        SingleRowGather(computeAddr, maxValueAddr, argmaxAddr);
    } else if (hOutputReal_ * wOutputReal_ * RATIO > vlT2_) {
        MultiRowGather(computeAddr, maxValueAddr, argmaxAddr);
    } else if( dOutputReal_ * hOutputReal_ * wOutputReal_ * RATIO > vlT2_){
        MultiDepGather(computeAddr, maxValueAddr, argmaxAddr);
    }else {
        MultiNcGather(computeAddr, maxValueAddr, argmaxAddr);
    }

    inputQue_.FreeTensor(inputLocal);
    maxValueQue_.EnQue(maxValueLocal);
    argmaxQue_.EnQue(argmaxLocal);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::ConvertIndexWithoutPadAlignNc(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,int32_t hInputActualPad, T2 left, T2 wInput, T2 hIndexBase,T2 hInput, T2 dIndexBase,
    MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize)
{
    MicroAPI::RegTensor<int32_t> ncIndexReg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(ncIndexReg, static_cast<int32_t>(0));
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(ncOutputCount));
    MicroAPI::Div(divResultReg, ncIndexReg, constReg, allMaskB32);
    MicroAPI::Muls(divResultReg, divResultReg, inputNcSize, allMaskB32);
    MicroAPI::Sub(srcReg, srcReg, divResultReg, allMaskB32);
    ConvertIndexWithoutPadAlign(srcReg, wStrideOffset, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                dstReg, ncInputOffset);
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::ConvertIndexWithoutPadAlign(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, uint32_t hInputActualPad, T2 left, T2 wInput, T2 hIndexBase, T2 hInput,T2 dIndexBase,
    MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset)
{
    MicroAPI::RegTensor<T2> hIndexReg;
    MicroAPI::RegTensor<T2> dIndexReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::RegTensor<int32_t> t3Reg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<int32_t> t1Reg;
    MicroAPI::RegTensor<int32_t> t2Reg;
    MicroAPI::RegTensor<T2> divResultRegUnpack;
    MicroAPI::RegTensor<T2> dIndexRegUnpack;
    MicroAPI::RegTensor<T2> wIndexReg;
    MicroAPI::RegTensor<int32_t> wIndexRegUnpack;
    MicroAPI::RegTensor<T2> zeroReg;
    MicroAPI::MaskReg negInfMask;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allMaskT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    MicroAPI::Duplicate(t3Reg, static_cast<int32_t>(wStrideOffset * hInputActualPad));
    MicroAPI::Duplicate(zeroReg, static_cast<T2>(0));
    MicroAPI::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    MicroAPI::Div(t1Reg, srcReg, t3Reg, allMaskB32);
    MicroAPI::Muls(t2Reg, t1Reg, static_cast<int32_t>(-1 * wStrideOffset * hInputActualPad), allMaskB32);
    MicroAPI::Add(t2Reg, srcReg, t2Reg, allMaskB32);
    MicroAPI::Div(divResultReg, t2Reg, constReg, allMaskB32);
    MicroAPI::Mul(t3Reg, divResultReg, constReg, allMaskB32);
    MicroAPI::Sub(wIndexRegUnpack, t2Reg, t3Reg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        MicroAPI::UnPack(dIndexRegUnpack, t1Reg);
        MicroAPI::UnPack(divResultRegUnpack, divResultReg);
        MicroAPI::UnPack(wIndexReg, wIndexRegUnpack);
        MicroAPI::Adds(dIndexReg, dIndexRegUnpack, dIndexBase, allMaskT2);
        MicroAPI::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
        MicroAPI::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        MicroAPI::Adds(dIndexReg, t1Reg, dIndexBase, allMaskB32);
        MicroAPI::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
        MicroAPI::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if constexpr (IS_PAD == 1) {
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, dIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(dIndexReg, zeroReg, dIndexReg, negInfMask);
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    MicroAPI::Muls(dIndexReg, dIndexReg, (wInput * hInput), allMaskT2);
    MicroAPI::Muls(hIndexReg, hIndexReg, wInput, allMaskT2);
    MicroAPI::Add(dstReg, hIndexReg, dIndexReg, allMaskT2);
    MicroAPI::Add(dstReg, dstReg, wIndexReg, allMaskT2);
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::ProcessW(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, int32_t hOffset, uint16_t wStrideOffset, uint16_t hInputActualPad,
    MicroAPI::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
    int32_t outputOffset, MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t dDilation ,uint32_t hDilation, uint32_t wDilation)
{
    MicroAPI::RegTensor<int32_t> indexWithOffset;
    MicroAPI::RegTensor<T1> calcReg;
    MicroAPI::RegTensor<int32_t> calcMaxIndexReg;
    uint32_t maskCount = repeatElem;
    MicroAPI::MaskReg allMaskU32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg gatherMask = MicroAPI::UpdateMask<T1>(maskCount);
    MicroAPI::RegTensor<T1> maxReg;
    MicroAPI::MaskReg neMask;
    MicroAPI::MaskReg gtMask;
    MicroAPI::MaskReg tmpMask;
    MicroAPI::UnalignReg u0;

    __local_mem__ T1* maxValueAddrLocal = maxValueAddr + outputOffset;
    SetNegInfReg<T1>(maxReg);
    MicroAPI::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);
    for (uint16_t d = 0; d < dKernel; d++) {
        for (uint16_t i = 0; i < hKernel; i++) {
            for (uint16_t j = 0; j < wKernel; j++) {
                int32_t relIndex = d * hInputActualPad * wStrideOffset * dDilation  + i * wStrideOffset * hDilation + j * wDilation;
                int32_t offset = static_cast<int32_t>(hOffset + relIndex);
                MicroAPI::Adds(indexWithOffset, indexReg, offset, allMaskU32);
                if constexpr (std::is_same<T1, float>::value) {
                    MicroAPI::DataCopyGather(calcReg, computeAddr, (MicroAPI::RegTensor<uint32_t>&)indexWithOffset,
                                            gatherMask);
                } else {
                    MicroAPI::RegTensor<uint16_t> indexConvert;
                    MicroAPI::Pack(indexConvert, indexWithOffset);
                    MicroAPI::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
                }
                MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
                MicroAPI::Compare<T1, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
                MicroAPI::MaskOr(gtMask, gtMask, neMask, gatherMask);
                if constexpr (sizeof(int32_t) / sizeof(T1) == 1) {
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
                } else {
                    MicroAPI::MaskUnPack(tmpMask, gtMask);
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, tmpMask);
                }
                MicroAPI::Max(maxReg, maxReg, calcReg, gatherMask);
            }
        }
    }
    MicroAPI::DataCopyUnAlign(maxValueAddrLocal, maxReg, u0, repeatElem);
    MicroAPI::DataCopyUnAlignPost(maxValueAddrLocal, u0, 0);
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::SingleRowGather(__local_mem__ T1* computeAddr,
                                                                                        __local_mem__ T1* maxValueAddr,
                                                                                        __local_mem__ T2* argmaxAddr)
{
    uint16_t loopW = wOutputReal_ / vlT2_;
    uint16_t repeatsElem = vlT2_;
    uint16_t tailRepeatsElem = wOutputReal_ - loopW * vlT2_;
    
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }

    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t wStride = tilingData_.wStride;
    T2 dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 wInput = tilingData_.wInput;
    T2 hInput = tilingData_.hInput;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t wOutputActual = wOutputReal_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t dDilation = tilingData_.dDilation;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++){
            for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
                __VEC_SCOPE__
                {
                    MicroAPI::RegTensor<int32_t> indexReg;
                    MicroAPI::RegTensor<int32_t> maxIndexReg;
                    MicroAPI::RegTensor<T2> maxIndexConvertReg;
                    MicroAPI::UnalignReg u1;
                    MicroAPI::Arange(indexReg, static_cast<int32_t>(0));
                    MicroAPI::MaskReg preg = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
                    MicroAPI::Muls(indexReg, indexReg, static_cast<int32_t>(wStride), preg);
                    int32_t ncInOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
                    int32_t ncOutOffset = nc * dOutputActual * hOutputActual * wOutputActual;
                    int32_t vfMaxAddrOffset = ncOutOffset +  dLoop * hOutputActual * wOutputActual  + hLoop * wOutputActual;
                    __local_mem__ T2* argmaxAddrLocal = argmaxAddr + vfMaxAddrOffset;
                    for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                        int32_t wOffset =
                            ncInOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad + hLoop * wInputActualAlignedPad * hStride 
                            + wLoop * repeatsElem * wStride;
                        int32_t wOutOffset =
                            ncOutOffset + dLoop * hOutputActual * wOutputActual  + hLoop * wOutputActual + wLoop * repeatsElem;
                        ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                                indexReg, dKernel ,hKernel, wKernel,repeatsElem, 
                                wOutOffset, maxIndexReg, dDilation, hDilation, wDilation);
                        ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                    maxIndexConvertReg, ncInOffset);
                        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                    }
                    int32_t wOffsetTail =
                            ncInOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad + hLoop * wInputActualAlignedPad * hStride 
                            + loopW * repeatsElem * wStride;
                    int32_t wOutOffsetTail =
                        ncOutOffset + dLoop * hOutputActual * wOutputActual + hLoop * wOutputActual + loopW * repeatsElem;
                    ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, indexReg, dKernel, hKernel, wKernel,
                            tailRepeatsElem, wOutOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0); 
                }
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::MultiRowGather(__local_mem__ T1* computeAddr,
                                                                                       __local_mem__ T1* maxValueAddr,
                                                                                       __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t wOutputActual = wOutputReal_;
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;

    uint16_t hBatchCount = vlT2_ / wOutputReal_;
    uint32_t rate2D = wInputActualAlignedPad_ * tilingData_.hStride;
    uint16_t hLoopTimes = hOutputReal_ / hBatchCount;
    uint16_t hTail = hOutputReal_ - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputReal_;
    uint16_t tailRepeatsElem = hTail * wOutputReal_;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    T2 dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    T2 hInput = tilingData_.hInput;
    T2 wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t dDilation = tilingData_.dDilation;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dOutputActual); dLoop++){
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<int32_t> indexReg;
                MicroAPI::RegTensor<int32_t> maxIndexReg;
                MicroAPI::RegTensor<T2> maxIndexConvertReg;
                MicroAPI::UnalignReg u1;
                __local_mem__ T2* argmaxAddrLocal;
                int32_t ncInputOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
                argmaxAddrLocal = argmaxAddr + nc * dOutputActual * hOutputActual * wOutputActual + dLoop * hOutputActual * wOutputActual;
                CalGatterIndex2D<int32_t>(indexReg, rate2D, wOutputActual, wStride);
                for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                    int32_t wOffset = ncInputOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad  +
                    hLoop * hBatchCount * hStride * wInputActualAlignedPad;
                    int32_t wOutputOffset = nc * dOutputActual * hOutputActual * wOutputActual + 
                        dLoop * hOutputActual * wOutputActual  + hLoop * hBatchCount * wOutputActual;
                    ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                                indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                                wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInputOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                }
                int32_t wOffsetTail = ncInputOffset + dLoop * dStride * hInputActualPad * wInputActualAlignedPad  +
                                            hLoopTimes * hBatchCount * hStride * wInputActualAlignedPad;
                int32_t wOutputOffsetTail = nc * dOutputActual * hOutputActual *+ wOutputActual + dLoop * hOutputActual * wOutputActual + 
                                            hLoopTimes * hBatchCount * wOutputActual;
                ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                                indexReg, dKernel ,hKernel, wKernel,tailRepeatsElem, 
                                wOutputOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                                maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::MultiDepGather(__local_mem__ T1* computeAddr,
                                                                                       __local_mem__ T1* maxValueAddr,
                                                                                       __local_mem__ T2* argmaxAddr)
{
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t dKernel = tilingData_.dKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t rate3D = tilingData_.dStride *  hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint16_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachDepCount = hOutputReal_ * wOutputReal_;
    uint16_t dBatchCount = vlT2_ / eachDepCount;
    uint16_t dLoopTimes = dOutputReal_ / dBatchCount;
    uint16_t dTail = dOutputReal_ - dLoopTimes * dBatchCount;
    if (dTail == 0) {
        dLoopTimes = dLoopTimes - 1;
        dTail = dBatchCount;
    }
    uint16_t repeatsElem = dBatchCount * eachDepCount;
    uint16_t tailRepeatsElem = dTail * eachDepCount;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    T2 dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    T2 hInput = tilingData_.hInput;
    T2 wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t dStride = tilingData_.dStride;
    uint32_t hStride = tilingData_.hStride;
    uint32_t dDilation = tilingData_.dDilation;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> indexReg;
            MicroAPI::RegTensor<int32_t> maxIndexReg;
            MicroAPI::RegTensor<T2> maxIndexConvertReg;
            MicroAPI::UnalignReg u1;
            __local_mem__ T2* argmaxAddrLocal = argmaxAddr;
            CalGatterIndex3D<int32_t>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
            argmaxAddrLocal = argmaxAddr + nc * dOutputActual * hOutputActual * wOutputActual;
            int32_t ncInputOffset = nc * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
            for(uint16_t dLoop = 0; dLoop < static_cast<uint16_t>(dLoopTimes); dLoop++){
                int32_t wOffset = ncInputOffset + dLoop * dBatchCount * dStride * hInputActualPad * wInputActualAlignedPad;
                int32_t wOutputOffset = nc * dOutputActual * hOutputActual * wOutputActual + 
                    dLoop * dBatchCount * hOutputActual * wOutputActual ;
                ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                            wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                            maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
            int32_t wOffsetTail = ncInputOffset + dLoopTimes * dBatchCount  *  dStride * hInputActualPad * wInputActualAlignedPad;
            int32_t wOutputOffsetTail = nc * dOutputActual * hOutputActual * wOutputActual + 
                    dLoopTimes * dBatchCount * hOutputActual * wOutputActual ;
            ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel ,hKernel, wKernel,tailRepeatsElem, 
                            wOutputOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
            ConvertIndexWithoutPadAlign(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                            maxIndexConvertReg, ncInputOffset);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::MultiNcGather(__local_mem__ T1* computeAddr,
                                                                                      __local_mem__ T1* maxValueAddr,
                                                                                      __local_mem__ T2* argmaxAddr)
{
    uint16_t dKernel = tilingData_.dKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t rate4D = dInputActualPad_ * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num3D = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint16_t rate3D = tilingData_.dStride *  hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputReal_ * wOutputReal_;
    uint16_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputReal_;
    uint16_t eachBatchCount = dOutputReal_ * hOutputReal_ * wOutputReal_;
    uint16_t ncBatchCount = vlT2_ / eachBatchCount;
    uint16_t ncLoopTimes = highAxisActual_ / ncBatchCount;
    uint16_t ncTail = highAxisActual_ - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    T2 dIndexBase = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.dStride - tilingData_.padFront;
    T2 hInput = tilingData_.hInput;
    T2 wInput = tilingData_.wInput;
    uint32_t dInputActualPad = dInputActualPad_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t dOutputActual = dOutputReal_;
    uint32_t hOutputActual = hOutputReal_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t dDilation = tilingData_.dDilation;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<T2> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ T2* argmaxAddrLocal = argmaxAddr;
        CalGatterIndex4D<int32_t>(indexReg, rate4D, num3D, rate3D, num2D, rate2D, wOutputActual, wStride);
        for (uint16_t nc = 0; nc < ncLoopTimes; nc++) {
            uint32_t wOffset = nc * ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
            uint32_t wOutputOffset = nc * ncBatchCount * dOutputActual *hOutputActual * wOutputActual;
            ProcessW(computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,repeatsElem, 
                            wOutputOffset, maxIndexReg, dDilation, hDilation, wDilation);
            ConvertIndexWithoutPadAlignNc(maxIndexReg, wInputActualAlignedPad, hInputActualPad,left, wInput, hIndexBase, hInput, dIndexBase,
                                          maxIndexConvertReg, wOffset, num3D, rate4D);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
        uint32_t wOffsetTail = ncLoopTimes * ncBatchCount * dInputActualPad * hInputActualPad * wInputActualAlignedPad;
        uint32_t wOutputOffsetTail = ncLoopTimes * ncBatchCount * dOutputActual* hOutputActual * wOutputActual;
        ProcessW(computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, hInputActualPad, 
                            indexReg, dKernel,hKernel, wKernel,tailRepeatsElem, 
                            wOutputOffsetTail, maxIndexReg, dDilation, hDilation, wDilation);
        ConvertIndexWithoutPadAlignNc(maxIndexReg, wInputActualAlignedPad, hInputActualPad, left, wInput, hIndexBase, hInput, dIndexBase,
                                          maxIndexConvertReg, wOffsetTail, num3D, rate4D);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPool3DWithArgmaxV2GatherKernel<T1, T2, IS_PAD>::CopyOut()
{
    LocalTensor<T1> maxValueLocal = maxValueQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    int64_t outPlaneSize = tilingData_.dOutput *tilingData_.hOutput * tilingData_.wOutput;
    int64_t highOutOffset = highAxisIndex_ * tilingData_.highAxisInner * outPlaneSize;
    int64_t dOutputOffset = dAxisIndex_ * tilingData_.dOutputInner * tilingData_.hOutput * tilingData_.wOutput;
    int64_t hOutputOffset = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.wOutput;
    int64_t wOutputOffset = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outGmOffset = highOutOffset + dOutputOffset + hOutputOffset + wOutputOffset;

    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(highAxisActual_ * dOutputReal_ * hOutputReal_ * wOutputReal_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outGmOffset], maxValueLocal, copyOutParamT1);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(highAxisActual_ * dOutputReal_ *hOutputReal_ * wOutputReal_ * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(argmaxGm_[outGmOffset], argmaxLocal, copyOutParamT2);
    maxValueQue_.FreeTensor(maxValueLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
    return;
}
}  // namespace MaxPool3DWithArgmaxV2GatherNameSpace
#endif