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
 * \file max_pool_with_argmax_v3_gather_kernel.h
 * \brief
 */
#ifndef MAX_POOL_WITH_ARGMAX_V3_GATHER_KERNEL_H_
#define MAX_POOL_WITH_ARGMAX_V3_GATHER_KERNEL_H_

#include "max_pool_with_argmax_v3_base.h"

namespace MaxPoolWithArgmaxV3GatherNameSpace {
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t HELPER_BUFFER_SIZE = 1024;
constexpr int64_t THREE_DIM = 3;
constexpr int64_t RATIO = 2;
constexpr uint16_t B32 = 4;

template <typename T1, typename T2, const uint32_t IS_PAD = 0>
class MaxPoolWithArgmaxV3GatherKernel {
public:
    __aicore__ inline MaxPoolWithArgmaxV3GatherKernel(
        TPipe& pipeIn, const MaxPoolWithArgmaxV3GatherTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void DupBufferNegInf(__local_mem__ T1* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail);
    __aicore__ inline void CopyToCalcBuffer(
        __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch, uint16_t rows, uint16_t loopCols,
        uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
        uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset);
    __aicore__ inline void DupAndCopyToCalcBuffer(__local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr);
    __aicore__ inline void ConvertIndexWithoutPadAlign(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
        MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset);
    __aicore__ inline void ConvertIndexWithoutPadAlignNc(
        MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
        MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize);
    __aicore__ inline void ProcessW(
        __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, int32_t hOffset, uint16_t wStrideOffset,
        MicroAPI::RegTensor<int32_t>& indexReg, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
        int32_t outputOffset, MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation, uint32_t wDilation);
    __aicore__ inline void SingleRowGather(
        __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiRowGather(
        __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);
    __aicore__ inline void MultiNcGather(
        __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr);

private:
    TPipe& pipe_;
    const MaxPoolWithArgmaxV3GatherTilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxValueQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;
    int64_t highAxisActual_ = 0;
    int64_t hOutputActual_ = 0;
    int64_t wOutputActual_ = 0;
    int64_t curCoreProcessNum_ = 0;
    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;
    int64_t wInputActualAlignedPad_ = 0;
    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t highInputAxisOffset_ = 0;
    int64_t hInputAxisOffset_ = 0;
    int64_t wInputAxisOffset_ = 0;

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
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
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
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;
    highAxisIndex_ = baseBlockIdx / (tilingData_.hOutputOuter * tilingData_.wOutputOuter);
    highAxisActual_ =
        highAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail : tilingData_.highAxisInner;
    int64_t tempTail = baseBlockIdx % (tilingData_.hOutputOuter * tilingData_.wOutputOuter);

    hAxisIndex_ = tempTail / tilingData_.wOutputOuter;
    hOutputActual_ = hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner;

    wAxisIndex_ = tempTail % tilingData_.wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner;
    wOutputActualAligned_ = CeilDivision(wOutputActual_, maxDataNumOneBlock_) * maxDataNumOneBlock_;

    hInputActualPad_ =
        (hOutputActual_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.hDilation + 1;
    wInputActualPad_ =
        (wOutputActual_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.wDilation + 1;

    wInputActualAlignedPad_ = CeilDivision(wInputActualPad_, blockSize_ / sizeof(T1)) * (blockSize_ / sizeof(T1));

    int64_t inputPlaneSize = tilingData_.hInput * tilingData_.wInput;
    highInputAxisOffset_ = highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    hInputAxisOffset_ = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride * tilingData_.wInput;
    wInputAxisOffset_ = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride;
    if constexpr (IS_PAD == 1) {
        int64_t tRelBoundaryDistance =
            hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;

        int64_t dRelBoundaryDistance = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride +
                                       (hOutputActual_ - 1) * tilingData_.hStride + tilingData_.hKernel -
                                       tilingData_.hInput - tilingData_.padTop;

        int64_t lRelBoundaryDistance =
            wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;

        int64_t rRelBoundaryDistance = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride +
                                       (wOutputActual_ - 1) * tilingData_.wStride + tilingData_.wKernel -
                                       tilingData_.wInput - tilingData_.padLeft;
        leftOffsetToInputLeft_ = lRelBoundaryDistance >= 0 ? 0 : -lRelBoundaryDistance;
        rightOffsetToInputRight_ = rRelBoundaryDistance >= 0 ? rRelBoundaryDistance : 0;
        topOffsetToInputTop_ = tRelBoundaryDistance >= 0 ? 0 : -tRelBoundaryDistance;
        downOffsetToInputDown_ = dRelBoundaryDistance >= 0 ? dRelBoundaryDistance : 0;

        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
        hInputAxisOffset_ = topOffsetToInputTop_ == 0 ? hInputAxisOffset_ - tilingData_.padTop * tilingData_.wInput : 0;

        wInputAxisOffset_ = leftOffsetToInputLeft_ == 0 ? wInputAxisOffset_ - tilingData_.padLeft : 0;
    }
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::ProcessPerLoop()
{
    CopyIn();
    Compute();
    CopyOut();
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::CopyIn()
{
    LocalTensor<T1> xLocal = inputQue_.AllocTensor<T1>();
    int64_t xGmOffset = highInputAxisOffset_ + hInputAxisOffset_ + wInputAxisOffset_;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = tilingData_.hInput * tilingData_.wInput * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = 0;

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
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::DupBufferNegInf(
    __local_mem__ T1* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    MicroAPI::RegTensor<T1> v0;
    DuplicateNegInfReg<T1>(v0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    uint32_t maskCount = tail;
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = MicroAPI::UpdateMask<T1>(maskCount);
    MicroAPI::DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::CopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr, uint16_t batch, uint16_t rows, uint16_t loopCols,
    uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
    uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    MicroAPI::RegTensor<T1> v0;
    MicroAPI::UnalignReg u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __local_mem__ T1* curSrcAddr = srcAddr + i * srcBatchStride + j * srcRowStride;
            __local_mem__ T1* curDstAddr =
                dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
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
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::DupAndCopyToCalcBuffer(
    __local_mem__ T1* dstAddr, __local_mem__ T1* srcAddr)
{
    uint16_t loopCols = wInputActualNoPad_ / vlT1_;
    uint16_t tailCols = wInputActualNoPad_ - loopCols * vlT1_;
    uint32_t wInputActualNoPadAlign =
        CeilDivision(wInputActualNoPad_, blockSize_ / sizeof(T1)) * blockSize_ / sizeof(T1);
    uint32_t dstBatchStride = hInputActualPad_ * wInputActualAlignedPad_;
    uint32_t totalInput = tilingData_.highAxisInner * hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t loopDup = totalInput / vlT1_;
    uint32_t tailDup = totalInput - loopDup * vlT1_;
    uint32_t dstRowOffset = topOffsetToInputTop_;
    uint32_t dstColOffset = leftOffsetToInputLeft_;
    __VEC_SCOPE__
    {
        DupBufferNegInf(dstAddr, vlT1_, loopDup, tailDup);
        CopyToCalcBuffer(
            dstAddr, srcAddr, highAxisActual_, hInputActualNoPad_, loopCols, tailCols, vlT1_, dstBatchStride,
            wInputActualNoPadAlign, dstBatchStride, wInputActualAlignedPad_, dstRowOffset, dstColOffset);
    }
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::Compute()
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
    if (wOutputActual_ * RATIO > vlT2_) {
        SingleRowGather(computeAddr, maxValueAddr, argmaxAddr);
    } else if (hOutputActual_ * wOutputActual_ * RATIO > vlT2_) {
        MultiRowGather(computeAddr, maxValueAddr, argmaxAddr);
    } else {
        MultiNcGather(computeAddr, maxValueAddr, argmaxAddr);
    }

    inputQue_.FreeTensor(inputLocal);
    maxValueQue_.EnQue(maxValueLocal);
    argmaxQue_.EnQue(argmaxLocal);
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::ConvertIndexWithoutPadAlignNc(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
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

    ConvertIndexWithoutPadAlign(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg, ncInputOffset);
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::ConvertIndexWithoutPadAlign(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
    MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset)
{
    MicroAPI::RegTensor<T2> hIndexReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<T2> divResultRegUnpack;
    MicroAPI::RegTensor<T2> wIndexReg;
    MicroAPI::RegTensor<int32_t> wIndexRegUnpack;
    MicroAPI::RegTensor<T2> zeroReg;
    MicroAPI::MaskReg negInfMask;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allMaskT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    MicroAPI::Duplicate(zeroReg, static_cast<T2>(0));
    MicroAPI::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    MicroAPI::Div(divResultReg, srcReg, constReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        MicroAPI::UnPack(divResultRegUnpack, divResultReg);
        MicroAPI::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
    } else {
        MicroAPI::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
    }
    if constexpr (IS_PAD == 1) {
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }
    MicroAPI::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskT2);
    MicroAPI::Mul(divResultReg, divResultReg, constReg, allMaskB32);
    MicroAPI::Sub(wIndexRegUnpack, srcReg, divResultReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        MicroAPI::UnPack(wIndexReg, wIndexRegUnpack);
        MicroAPI::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        MicroAPI::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if constexpr (IS_PAD == 1) {
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    MicroAPI::Add(dstReg, hIndexReg, wIndexReg, allMaskT2);
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::ProcessW(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, int32_t hOffset, uint16_t wStrideOffset,
    MicroAPI::RegTensor<int32_t>& indexReg, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
    int32_t outputOffset, MicroAPI::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation, uint32_t wDilation)
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
    DuplicateNegInfReg<T1>(maxReg);
    MicroAPI::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);
    for (uint16_t i = 0; i < hKernel; i++) {
        for (uint16_t j = 0; j < wKernel; j++) {
            int32_t relIndex = i * wStrideOffset * hDilation + j * wDilation;
            int32_t offset = static_cast<int32_t>(hOffset + relIndex);
            MicroAPI::Adds(indexWithOffset, indexReg, offset, allMaskU32);
            if constexpr (std::is_same<T1, float>::value) {
                MicroAPI::DataCopyGather(
                    calcReg, computeAddr, (MicroAPI::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
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
    MicroAPI::DataCopyUnAlign(maxValueAddrLocal, maxReg, u0, repeatElem);
    MicroAPI::DataCopyUnAlignPost(maxValueAddrLocal, u0, 0);
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::SingleRowGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t loopW = wOutputActual_ / vlT2_;
    uint16_t repeatsElem = vlT2_;
    uint16_t tailRepeatsElem = wOutputActual_ - loopW * vlT2_;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }
    uint16_t hKernel = tilingData_.hKernel;
    uint16_t wKernel = tilingData_.wKernel;
    uint32_t hStride = tilingData_.hStride;
    uint32_t wStride = tilingData_.wStride;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    T2 wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t hOutputActual = hOutputActual_;
    uint32_t wOutputActual = wOutputActual_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> indexReg;
            MicroAPI::RegTensor<int32_t> maxIndexReg;
            MicroAPI::RegTensor<T2> maxIndexConvertReg;
            MicroAPI::UnalignReg u1;
            MicroAPI::Arange(indexReg, static_cast<int32_t>(0));
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Muls(indexReg, indexReg, static_cast<int32_t>(wStride), preg);
            int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
            int32_t ncOutputOffset = nc * hOutputActual * wOutputActual;
            __local_mem__ T2* argmaxAddrLocal = argmaxAddr + ncOutputOffset;
            for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
                for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                    int32_t wOffset =
                        ncInputOffset + hLoop * wInputActualAlignedPad * hStride + wLoop * repeatsElem * wStride;
                    int32_t wOutputOffset = ncOutputOffset + hLoop * wOutputActual + wLoop * repeatsElem;
                    ProcessW(
                        computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, indexReg, hKernel, wKernel,
                        repeatsElem, wOutputOffset, maxIndexReg, hDilation, wDilation);
                    ConvertIndexWithoutPadAlign(
                        maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg,
                        ncInputOffset);
                    MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                    MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
                }
                int32_t wOffsetTail =
                    ncInputOffset + hLoop * wInputActualAlignedPad * hStride + loopW * repeatsElem * wStride;
                int32_t wOutputOffsetTail = ncOutputOffset + hLoop * wOutputActual + loopW * repeatsElem;
                ProcessW(
                    computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, indexReg, hKernel, wKernel,
                    tailRepeatsElem, wOutputOffsetTail, maxIndexReg, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(
                    maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
        }
    }
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::MultiRowGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint32_t wOutputActual = wOutputActual_;
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint32_t wStride = tilingData_.wStride;
    uint32_t rate2D = wInputActualAlignedPad_ * tilingData_.hStride;
    uint16_t hBatchCount = vlT2_ / wOutputActual_;
    uint16_t hLoopTimes = hOutputActual_ / hBatchCount;
    uint16_t hTail = hOutputActual_ - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputActual_;
    uint16_t tailRepeatsElem = hTail * wOutputActual_;
    T2 left = wAxisIndex_ * tilingData_.wOutputInner * tilingData_.wStride - tilingData_.padLeft;
    T2 hIndexBase = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.hStride - tilingData_.padTop;
    T2 wInput = tilingData_.wInput;
    uint32_t highAxisActual = highAxisActual_;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t hOutputActual = hOutputActual_;
    uint32_t hStride = tilingData_.hStride;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<T2> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ T2* argmaxAddrLocal = argmaxAddr;
        GenGatterIndex2D<int32_t>(indexReg, rate2D, wOutputActual, wStride);
        for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
            int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
            for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                int32_t wOffset = ncInputOffset + hLoop * hBatchCount * hStride * wInputActualAlignedPad;
                int32_t wOutputOffset = nc * hOutputActual * wOutputActual + hLoop * hBatchCount * wOutputActual;
                ProcessW(
                    computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, indexReg, hKernel, wKernel, repeatsElem,
                    wOutputOffset, maxIndexReg, hDilation, wDilation);
                ConvertIndexWithoutPadAlign(
                    maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg, ncInputOffset);
                MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
            int32_t wOffsetTail = ncInputOffset + hLoopTimes * hBatchCount * hStride * wInputActualAlignedPad;
            int32_t wOutputOffsetTail = nc * hOutputActual * wOutputActual + hLoopTimes * hBatchCount * wOutputActual;
            ProcessW(
                computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, indexReg, hKernel, wKernel,
                tailRepeatsElem, wOutputOffsetTail, maxIndexReg, hDilation, wDilation);
            ConvertIndexWithoutPadAlign(
                maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg, ncInputOffset);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    }
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::MultiNcGather(
    __local_mem__ T1* computeAddr, __local_mem__ T1* maxValueAddr, __local_mem__ T2* argmaxAddr)
{
    uint16_t wKernel = tilingData_.wKernel;
    uint16_t hKernel = tilingData_.hKernel;
    uint32_t wStride = tilingData_.wStride;
    uint16_t rate3D = hInputActualPad_ * wInputActualAlignedPad_;
    uint16_t num2D = hOutputActual_ * wOutputActual_;
    uint16_t rate2D = tilingData_.hStride * wInputActualAlignedPad_;
    uint16_t wOutputActual = wOutputActual_;
    uint16_t eachBatchCount = hOutputActual_ * wOutputActual_;
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
    T2 wInput = tilingData_.wInput;
    uint32_t hInputActualPad = hInputActualPad_;
    uint32_t wInputActualAlignedPad = wInputActualAlignedPad_;
    uint32_t hOutputActual = hOutputActual_;
    uint32_t wOutputActualAligned = wOutputActualAligned_;
    uint32_t hDilation = tilingData_.hDilation;
    uint32_t wDilation = tilingData_.wDilation;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<T2> maxIndexConvertReg;
        MicroAPI::UnalignReg u1;
        __local_mem__ T2* argmaxAddrLocal = argmaxAddr;
        GenGatterIndex3D<int32_t>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);
        for (uint16_t nc = 0; nc < ncLoopTimes; nc++) {
            uint32_t wOffset = nc * ncBatchCount * hInputActualPad * wInputActualAlignedPad;
            uint32_t wOutputOffset = nc * ncBatchCount * hOutputActual * wOutputActual;
            ProcessW(
                computeAddr, maxValueAddr, wOffset, wInputActualAlignedPad, indexReg, hKernel, wKernel, repeatsElem,
                wOutputOffset, maxIndexReg, hDilation, wDilation);
            ConvertIndexWithoutPadAlignNc(
                maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg, wOffset, num2D,
                rate3D);
            MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
            MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
        uint32_t wOffsetTail = ncLoopTimes * ncBatchCount * hInputActualPad * wInputActualAlignedPad;
        uint32_t wOutputOffsetTail = ncLoopTimes * ncBatchCount * hOutputActual * wOutputActual;
        ProcessW(
            computeAddr, maxValueAddr, wOffsetTail, wInputActualAlignedPad, indexReg, hKernel, wKernel, tailRepeatsElem,
            wOutputOffsetTail, maxIndexReg, hDilation, wDilation);
        ConvertIndexWithoutPadAlignNc(
            maxIndexReg, wInputActualAlignedPad, left, wInput, hIndexBase, maxIndexConvertReg, wOffsetTail, num2D,
            rate3D);
        MicroAPI::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
        MicroAPI::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
    return;
}
template <typename T1, typename T2, const uint32_t IS_PAD>
__aicore__ inline void MaxPoolWithArgmaxV3GatherKernel<T1, T2, IS_PAD>::CopyOut()
{
    LocalTensor<T1> maxValueLocal = maxValueQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    int64_t outputPlaneSize = tilingData_.hOutput * tilingData_.wOutput;
    int64_t highOutputAxisOffset = highAxisIndex_ * tilingData_.highAxisInner * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_.hOutputInner * tilingData_.wOutput;
    int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(highAxisActual_ * hOutputActual_ * wOutputActual_ * sizeof(T1)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], maxValueLocal, copyOutParamT1);
    DataCopyExtParams copyOutParamT2 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(highAxisActual_ * hOutputActual_ * wOutputActual_ * sizeof(T2)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(argmaxGm_[outputGmOffset], argmaxLocal, copyOutParamT2);
    maxValueQue_.FreeTensor(maxValueLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
    return;
}
} // namespace MaxPoolWithArgmaxV3GatherNameSpace
#endif