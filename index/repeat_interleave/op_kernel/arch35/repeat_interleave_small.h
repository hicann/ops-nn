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
 * \file repeat_interleave_small.h
 * \brief
 */

#ifndef REPEAT_INTERLEAVE_SMALL_H
#define REPEAT_INTERLEAVE_SMALL_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint64_t DOUBLE = 1;
constexpr uint64_t MERGED_DIM_LENGTH = 3;
constexpr int64_t OFFSET_BUFFER_LENGTH = 32;
constexpr uint64_t UB_ALIGN_VALUE = 32;
constexpr uint64_t CP_THRESHOLD = 256;
constexpr uint64_t CACHELINE_SIZE = 128;
constexpr uint64_t SPLIT_COMPUTE_THRESHOLD = 16384;

template <typename T, typename U>
class RepeatInterleaveSmall {
public:
    __aicore__ inline RepeatInterleaveSmall(TPipe& pipe, const RepeatInterleaveTilingKernelDataSmall& tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t repeatCount, int64_t inputOffset);
    __aicore__ inline void CopyInAlign(int64_t repeatCount, int64_t inputOffset);
    __aicore__ inline void IndexRepeat(
        __ubuf__ T* inputAddr, __ubuf__ T* outputAddr, uint16_t repeatTimes, int32_t inputStride);
    __aicore__ inline void IndexRepeatGroup(
        __ubuf__ T* inputAddr, __ubuf__ T* outputAddr, const LocalTensor<U>& repeatsLocal, int32_t repeatLen,
        int32_t inputStride);
    __aicore__ inline void CopyOut(int64_t repeatCount);
    __aicore__ inline void CopyInRepeats(int64_t repeatCount, int64_t offset);
    __aicore__ inline int64_t ComputeOutputOffset(uint16_t repeatCount);
    __aicore__ inline int64_t CalcRepeatsPartStart(int64_t repeatCount);
    __aicore__ inline int64_t CalcOutputStartOfset();
    __aicore__ inline void CustomReduceSum(const LocalTensor<U>& dst, const LocalTensor<U>& src, uint16_t dataLen);
    __aicore__ inline void IndexRepeatLarge(
        __ubuf__ T* inputAddr, int64_t repeatTimes, int64_t outputRepeatsSum, int64_t outputOffset,
        int64_t repeatsIndex);
    __aicore__ inline void CopyXToOut(int64_t repeatCount);
    __aicore__ inline void CopyOutY(int64_t repeatCount);
    __aicore__ inline void CopyOutData(int64_t repeatSum, int64_t outputOffset);
    __aicore__ inline void CopyOutSplit(int64_t repeatCount);
    __aicore__ inline void CopyOutSplitImpl(
        __ubuf__ T* inputAddr, int64_t outputOffset, const LocalTensor<U>& repeatsLocal, int64_t repeatCount);
    __aicore__ inline void CopyOutSplitGroup(
        __ubuf__ T* inputAddr, int64_t outputOffset, const LocalTensor<U>& repeatsLocal, int64_t repeatCount);

private:
    TPipe& pipe_;
    const RepeatInterleaveTilingKernelDataSmall& tilingData_;
    GlobalTensor<T> xGm_;
    GlobalTensor<U> repeatsGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<U> recordSumGm_;

    TQue<QuePosition::VECIN, DOUBLE> inputQueue_;
    TQue<QuePosition::VECOUT, DOUBLE> outputQueue_;
    TQue<QuePosition::VECIN, DOUBLE> repeatsQueue_;
    TQue<QuePosition::VECIN, DOUBLE> outputOffsetBuf_;
    TQue<QuePosition::VECIN, DOUBLE> recordOffsetBuf_;

    int64_t blockIdx_ = 0;
    int64_t currentRepeatCount_ = 0;
    int64_t outputOffsetRepeat_ = 0;
    int64_t currentRepeatsSum_ = 0;
    int64_t repeatsFactorSum_ = 0;
    int64_t repeatsFactorAlign_ = 0;
    int64_t cpAlign_ = 0;
    int64_t loopOfset_ = 0;
    int64_t batchNO_ = 0;
    int64_t sliceNO_ = 0;
    GM_ADDR repeats_ = 0;
};

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace)
{
    pipe_.InitBuffer(repeatsQueue_, DOUBLE, tilingData_.ubFactor * sizeof(U));
    pipe_.InitBuffer(inputQueue_, DOUBLE, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(outputQueue_, DOUBLE, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(outputOffsetBuf_, DOUBLE, OFFSET_BUFFER_LENGTH * sizeof(U));
    pipe_.InitBuffer(recordOffsetBuf_, DOUBLE, GetBlockNum() * UB_ALIGN_VALUE);

    blockIdx_ = GetBlockIdx();
    repeats_ = repeats;
    batchNO_ = blockIdx_ / tilingData_.repeatsSlice;
    sliceNO_ = blockIdx_ % tilingData_.repeatsSlice;
    int64_t inputOffset = batchNO_ * tilingData_.mergedDims[1] * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] +
                          sliceNO_ * tilingData_.normalRepeatsCount * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    xGm_.SetGlobalBuffer((__gm__ T*)x + inputOffset);
    repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    recordSumGm_.SetGlobalBuffer((__gm__ U*)workspace);

    if (sliceNO_ == tilingData_.repeatsSlice - 1) {
        currentRepeatCount_ = tilingData_.tailRepeatsCount;
    } else {
        currentRepeatCount_ = tilingData_.normalRepeatsCount;
    }
    repeatsFactorSum_ = tilingData_.ubFactor / tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    cpAlign_ =
        Ops::Base::CeilAlign(tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] * sizeof(T), UB_ALIGN_VALUE) / sizeof(T);
    repeatsFactorAlign_ = tilingData_.ubFactor / cpAlign_;
}
template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyIn(int64_t repeatCount, int64_t inputOffset)
{
    LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = repeatCount * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(
        inputLocal, xGm_[inputOffset * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]], dataCopyParams, padParams);
    inputQueue_.EnQue(inputLocal);

    CopyInRepeats(repeatCount, inputOffset);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyInAlign(int64_t repeatCount, int64_t inputOffset)
{
    LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams = {true, 0, static_cast<uint8_t>(cpAlign_ - tilingData_.mergedDims[2]), 0};

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = repeatCount;
    dataCopyParams.blockLen = tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(
        inputLocal, xGm_[inputOffset * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]], dataCopyParams, padParams);
    inputQueue_.EnQue(inputLocal);
    CopyInRepeats(repeatCount, inputOffset);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::IndexRepeatLarge(
    __ubuf__ T* inputAddr, int64_t repeatTimes, int64_t outputRepeatsSum, int64_t outputOffset, int64_t repeatsIndex)
{
    int64_t loopCount = Ops::Base::CeilDiv(repeatTimes, outputRepeatsSum);
    uint16_t tailLoopTimes = repeatTimes - (loopCount - 1) * outputRepeatsSum;
    int64_t outputOffsetLocal = outputOffset;
    LocalTensor<T> outputLocal;
    __ubuf__ T* outLocalAddr;
    auto inputAddrLocal = inputAddr;
    for (int64_t i = 0; i < loopCount - 1; i++) {
        outputLocal = outputQueue_.AllocTensor<T>();
        outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
        IndexRepeat(
            inputAddrLocal, outLocalAddr, static_cast<uint16_t>(outputRepeatsSum),
            tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
        outputQueue_.EnQue(outputLocal);
        CopyOutData(outputRepeatsSum, outputOffsetLocal);
        outputOffsetLocal += outputRepeatsSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    }

    outputLocal = outputQueue_.AllocTensor<T>();
    outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
    IndexRepeat(inputAddrLocal, outLocalAddr, tailLoopTimes, tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
    outputQueue_.EnQue(outputLocal);
    CopyOutData(tailLoopTimes, outputOffsetLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::IndexRepeat(
    __ubuf__ T* inputAddr, __ubuf__ T* outputAddr, uint16_t repeatTimes, int32_t inputStride)
{
    __ubuf__ T* inputAddrLocal = inputAddr;
    __ubuf__ T* outputAddrLocal = outputAddr;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<T> inputRegTensor;
        AscendC::MicroAPI::DataCopyUnAlignPre(uIn, inputAddrLocal);
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            inputRegTensor, uIn, inputAddrLocal, inputStride);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                outputAddrLocal, inputRegTensor, uOut, inputStride);
            AscendC::MicroAPI::DataCopyUnAlignPost(outputAddrLocal, uOut, 0);
        }
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::IndexRepeatGroup(
    __ubuf__ T* inputAddr, __ubuf__ T* outputAddr, const LocalTensor<U>& repeatsLocal, int32_t repeatLen,
    int32_t inputStride)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<T> inputRegTensor;
        for (uint16_t repeatIdx = 0; repeatIdx < static_cast<uint16_t>(repeatLen); repeatIdx++) {
            uint16_t repeatTimes = repeatsLocal.GetValue(repeatIdx);
            AscendC::MicroAPI::DataCopyUnAlignPre(uIn, inputAddr);
            AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                inputRegTensor, uIn, inputAddr, inputStride);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outputAddr, inputRegTensor, uOut, inputStride);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(outputAddr, uOut, 0);
        }
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOutData(int64_t repeatSum, int64_t outputOffset)
{
    LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
    DataCopyExtParams copyOutParamData;
    copyOutParamData.blockCount = 1;
    copyOutParamData.blockLen = repeatSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] * sizeof(T);
    copyOutParamData.srcStride = 0;
    copyOutParamData.dstStride = 0;

    DataCopyPad(yGm_[outputOffset], outputLocal, copyOutParamData);
    outputQueue_.FreeTensor(outputLocal);

    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOutSplitImpl(
    __ubuf__ T* inputAddr, int64_t outputOffset, const LocalTensor<U>& repeatsLocal, int64_t repeatCount)
{
    LocalTensor<T> outputLocal;
    __ubuf__ T* outLocalAddr;
    int64_t repeatsSum = 0;
    int64_t repeatsSumTotal = 0;
    for (int64_t i = 0; i < repeatCount; i++) {
        int64_t repeatTimes = repeatsLocal.GetValue(i);
        repeatsSumTotal += repeatTimes;
        if (repeatTimes > repeatsFactorSum_) {
            if (repeatsSum > 0) {
                outputQueue_.EnQue(outputLocal);
                CopyOutData(repeatsSum, outputOffset);
                outputOffset += repeatsSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
                repeatsSum = 0;
            }
            IndexRepeatLarge(inputAddr, repeatTimes, repeatsFactorSum_, outputOffset, i);
            outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
            outputOffset += repeatTimes * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
        } else {
            if (repeatsSum == 0) {
                outputLocal = outputQueue_.AllocTensor<T>();
                outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
            }
            repeatsSum += repeatTimes;
            if (repeatsSum > repeatsFactorSum_) {
                outputQueue_.EnQue(outputLocal);
                CopyOutData(repeatsSum - repeatTimes, outputOffset);
                outputOffset += (repeatsSum - repeatTimes) * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
                repeatsSum = repeatTimes;
                outputLocal = outputQueue_.AllocTensor<T>();
                outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
            }
            IndexRepeat(
                inputAddr, outLocalAddr, static_cast<uint16_t>(repeatTimes),
                tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
            outLocalAddr = outLocalAddr + repeatTimes * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
        }
        inputAddr = inputAddr + tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    }
    if (repeatsSum > 0) {
        outputQueue_.EnQue(outputLocal);
        CopyOutData(repeatsSum, outputOffset);
    }
    outputOffsetRepeat_ += repeatsSumTotal;
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOutSplitGroup(
    __ubuf__ T* inputAddr, int64_t outputOffset, const LocalTensor<U>& repeatsLocal, int64_t repeatCount)
{
    int64_t repeatsSum = 0;

    LocalTensor<U> sumOutputLocal = outputOffsetBuf_.AllocTensor<U>();
    int64_t mainLen = Ops::Base::CeilDiv(repeatCount, tilingData_.averageRepeatTime);
    mainLen = Ops::Base::CeilAlign(mainLen * sizeof(U), UB_ALIGN_VALUE) / sizeof(U);
    int64_t loopSize = Ops::Base::CeilDiv(repeatCount, mainLen);
    int64_t tailLen = repeatCount - (loopSize - 1) * mainLen;

    auto inputAddrStride = mainLen * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    for (int64_t i = 0; i < loopSize; i++) {
        auto handleRepeatCount = (i == loopSize - 1) ? tailLen : mainLen;
        auto inputStartAddr = inputAddr + (i * inputAddrStride);
        CustomReduceSum(sumOutputLocal, repeatsLocal[i * mainLen], handleRepeatCount);
        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        repeatsSum = sumOutputLocal.GetValue(0);

        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        auto outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
        if (repeatsSum <= repeatsFactorSum_) {
            IndexRepeatGroup(
                inputStartAddr, outLocalAddr, repeatsLocal[i * mainLen], handleRepeatCount,
                tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
            outputQueue_.EnQue(outputLocal);
            CopyOutData(repeatsSum, outputOffset);
            outputOffsetRepeat_ += repeatsSum;
        } else {
            CopyOutSplitImpl(inputStartAddr, outputOffset, repeatsLocal[i * mainLen], handleRepeatCount);
        }
        outputOffset += (repeatsSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
    }
    outputOffsetBuf_.FreeTensor(sumOutputLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOutSplit(int64_t repeatCount)
{
    LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    int64_t outputOffset = batchNO_ * tilingData_.totalRepeatSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] +
                           outputOffsetRepeat_ * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    auto inputAddr = reinterpret_cast<__ubuf__ T*>(inputLocal[0].GetPhyAddr());
    event_t eventMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventMTE2ToS);
    CopyOutSplitGroup(inputAddr, outputOffset, repeatsLocal, repeatCount);
    inputQueue_.FreeTensor(inputLocal);
    repeatsQueue_.FreeTensor(repeatsLocal);
    return;
}
template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOut(int64_t repeatCount)
{
    LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();

    int64_t outputOffset = batchNO_ * tilingData_.totalRepeatSum * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1] +
                           outputOffsetRepeat_ * tilingData_.mergedDims[MERGED_DIM_LENGTH - 1];
    auto outLocalAddr = reinterpret_cast<__ubuf__ T*>(outputLocal[0].GetPhyAddr());
    auto inputAddr = reinterpret_cast<__ubuf__ T*>(inputLocal[0].GetPhyAddr());
    event_t eventMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventMTE2ToS);
    IndexRepeatGroup(inputAddr, outLocalAddr, repeatsLocal, repeatCount, tilingData_.mergedDims[MERGED_DIM_LENGTH - 1]);
    outputQueue_.EnQue(outputLocal);
    CopyOutData(currentRepeatsSum_, outputOffset);

    inputQueue_.FreeTensor(inputLocal);
    repeatsQueue_.FreeTensor(repeatsLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyXToOut(int64_t repeatCount)
{
    LocalTensor<T> xInLocal = inputQueue_.DeQue<T>();
    LocalTensor<T> xOutLocal = outputQueue_.AllocTensor<T>();

    __local_mem__ int8_t* xInLocalPtr = (__local_mem__ int8_t*)xInLocal.GetPhyAddr();
    __local_mem__ int8_t* xOutLocalPtr = (__local_mem__ int8_t*)xOutLocal.GetPhyAddr();

    uint32_t totalBytes = repeatCount * cpAlign_ * sizeof(T);
    uint16_t stride = Ops::Base::GetVRegSize();
    uint16_t size = (totalBytes + stride - 1) / stride;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
        uint32_t sreg = totalBytes;
        AscendC::MicroAPI::MaskReg preg;

        for (uint16_t i = 0; i < size; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<int8_t>(i, stride);
            AscendC::MicroAPI::DataCopy(inputRegTensor, xInLocalPtr, offset);
            AscendC::MicroAPI::DataCopy(xOutLocalPtr, inputRegTensor, offset, preg);
        }
    }
    outputQueue_.EnQue(xOutLocal);
    inputQueue_.FreeTensor(xInLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyOutY(int64_t repeatCount)
{
    int64_t outputOffset = (batchNO_ * tilingData_.totalRepeatSum + outputOffsetRepeat_) * tilingData_.mergedDims[2];

    CopyXToOut(repeatCount);
    LocalTensor<T> xOutLocal = outputQueue_.DeQue<T>();
    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    for (int32_t i = 0; i < repeatCount; i++) {
        uint16_t repeatTimes = static_cast<uint16_t>(repeatsLocal.GetValue(i));
        LoopModeParams loopParams;
        loopParams.loop1Size = repeatTimes;
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = 0;
        loopParams.loop2SrcStride = 0;
        loopParams.loop1DstStride = tilingData_.mergedDims[2] * sizeof(T);
        loopParams.loop2DstStride = 0;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = tilingData_.mergedDims[2] * sizeof(T);
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(yGm_[outputOffset + loopOfset_], xOutLocal[i * cpAlign_], outParams);
        loopOfset_ += repeatTimes * tilingData_.mergedDims[2];
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    outputQueue_.FreeTensor(xOutLocal);
    repeatsQueue_.FreeTensor(repeatsLocal);
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CopyInRepeats(int64_t repeatCount, int64_t offset)
{
    LocalTensor<U> repeatsLocal = repeatsQueue_.AllocTensor<U>();
    DataCopyPadExtParams<U> padParams = {false, 0, 0, 0};
    DataCopyExtParams dataCopyParamsRepeats;
    dataCopyParamsRepeats.blockCount = 1;
    dataCopyParamsRepeats.blockLen = repeatCount * sizeof(U);
    dataCopyParamsRepeats.srcStride = 0;
    dataCopyParamsRepeats.dstStride = 0;
    DataCopyPad(repeatsLocal, repeatsGm_[offset], dataCopyParamsRepeats, padParams);

    repeatsQueue_.EnQue(repeatsLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::CustomReduceSum(
    const LocalTensor<U>& dstLocal, const LocalTensor<U>& src, uint16_t dataLen)
{
    uint16_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t loopSize = (dataLen + vfLen - 1) / vfLen;
    auto srcAddr = (__ubuf__ U*)src.GetPhyAddr();
    auto dstAddr = (__ubuf__ U*)dstLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<U> src;
        AscendC::MicroAPI::RegTensor<U> dst;
        AscendC::MicroAPI::RegTensor<U> tmpSum;
        uint32_t pnum = static_cast<uint32_t>(dataLen);
        uint32_t sumMask = 1;
        AscendC::MicroAPI::MaskReg oneMask = AscendC::MicroAPI::UpdateMask<U>(sumMask);
        AscendC::MicroAPI::Duplicate(dst, static_cast<U>(0), oneMask);
        for (uint16_t i = 0; i < loopSize; i++) {
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<U>(pnum);
            AscendC::MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(src, srcAddr, vfLen);
            AscendC::MicroAPI::ReduceSum(tmpSum, src, pMask);
            AscendC::MicroAPI::Add(dst, dst, tmpSum, oneMask);
        }
        AscendC::MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, dst, 0, oneMask);
    }
}
template <typename T, typename U>
__aicore__ inline int64_t RepeatInterleaveSmall<T, U>::ComputeOutputOffset(uint16_t repeatCount)
{
    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    LocalTensor<U> sumOutputLocal = outputOffsetBuf_.AllocTensor<U>();

    CustomReduceSum(sumOutputLocal, repeatsLocal, repeatCount);
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    int64_t result = sumOutputLocal.GetValue(0);
    repeatsQueue_.FreeTensor(repeatsLocal);
    outputOffsetBuf_.FreeTensor(sumOutputLocal);
    return result;
}

template <typename T, typename U>
__aicore__ inline int64_t RepeatInterleaveSmall<T, U>::CalcRepeatsPartStart(int64_t repeatCount)
{
    int64_t result = 0;
    if (repeatCount == 0) {
        return result;
    }

    if (repeatCount <= tilingData_.ubFactor) {
        CopyInRepeats(repeatCount, 0);
        result = ComputeOutputOffset(static_cast<uint16_t>(repeatCount));
        return result;
    }
    int64_t loopCount = Ops::Base::CeilDiv(repeatCount, tilingData_.ubFactor);
    uint16_t remainCount = repeatCount - (loopCount - 1) * tilingData_.ubFactor;
    for (int64_t i = 0; i < loopCount - 1; i++) {
        CopyInRepeats(tilingData_.ubFactor, i * tilingData_.ubFactor);
        result += ComputeOutputOffset(static_cast<uint16_t>(tilingData_.ubFactor));
    }
    CopyInRepeats(remainCount, (loopCount - 1) * tilingData_.ubFactor);
    result += ComputeOutputOffset(remainCount);
    return result;
}

template <typename T, typename U>
__aicore__ inline int64_t RepeatInterleaveSmall<T, U>::CalcOutputStartOfset()
{
    int64_t cacheLineNum = CACHELINE_SIZE / sizeof(U);
    recordSumGm_(blockIdx_ * cacheLineNum) = currentRepeatsSum_;
    AscendC::DataCacheCleanAndInvalid<U, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        recordSumGm_[blockIdx_ * cacheLineNum]);
    SyncAll();

    uint8_t alignSize = UB_ALIGN_VALUE / sizeof(U);
    LocalTensor<U> recordOfsetLocal = recordOffsetBuf_.AllocTensor<U>();
    LocalTensor<U> sumOutputLocal = outputOffsetBuf_.AllocTensor<U>();
    DataCopyExtParams inParams = {
        static_cast<uint16_t>(tilingData_.usedCoreNum), static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>((cacheLineNum - 1) * sizeof(U)), 0, 0};
    DataCopyPadExtParams<U> padParams = {true, 0, static_cast<uint8_t>(alignSize - 1), 0};
    DataCopyPad(recordOfsetLocal, recordSumGm_, inParams, padParams);

    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    /* 计算从当前核所在的batch起始的repeats和 */
    auto batchStartOfset = batchNO_ * tilingData_.repeatsSlice * alignSize;
    CustomReduceSum(sumOutputLocal, recordOfsetLocal[batchStartOfset], blockIdx_ * alignSize - batchStartOfset);
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    int64_t outputStartOfset = sumOutputLocal.GetValue(0);

    outputOffsetBuf_.FreeTensor(sumOutputLocal);
    recordOffsetBuf_.FreeTensor(recordOfsetLocal);

    return outputStartOfset;
}

template <typename T, typename U>
__aicore__ inline void RepeatInterleaveSmall<T, U>::Process()
{
    if ((blockIdx_ + 1) > tilingData_.usedCoreNum) {
        return;
    }

    if (tilingData_.mergedDims[1] * sizeof(U) >= SPLIT_COMPUTE_THRESHOLD) {
        /* repeats大小超过阈值时分核累加计算outofset */
        repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats_ + sliceNO_ * tilingData_.normalRepeatsCount);
        currentRepeatsSum_ = CalcRepeatsPartStart(currentRepeatCount_);
        outputOffsetRepeat_ = CalcOutputStartOfset();
    } else {
        /* 单核计算outofset */
        outputOffsetRepeat_ = CalcRepeatsPartStart(sliceNO_ * tilingData_.normalRepeatsCount);
        repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats_ + sliceNO_ * tilingData_.normalRepeatsCount);
        currentRepeatsSum_ = CalcRepeatsPartStart(currentRepeatCount_);
    }

    if (tilingData_.mergedDims[2] * sizeof(T) > CP_THRESHOLD) {
        int64_t loop = Ops::Base::CeilDiv(currentRepeatCount_, repeatsFactorAlign_);
        int64_t tailLoopCount = currentRepeatCount_ - (loop - 1) * repeatsFactorAlign_;
        for (int64_t i = 0; i < loop - 1; i++) {
            CopyInAlign(repeatsFactorAlign_, i * repeatsFactorAlign_);
            CopyOutY(repeatsFactorAlign_);
        }
        CopyInAlign(tailLoopCount, (loop - 1) * repeatsFactorAlign_);
        CopyOutY(tailLoopCount);
        return;
    }

    // output可以一次性搬入
    if ((currentRepeatsSum_ <= repeatsFactorSum_) && (currentRepeatCount_ <= repeatsFactorSum_)) {
        CopyIn(currentRepeatCount_, 0);
        CopyOut(currentRepeatCount_);
    } else if (currentRepeatCount_ <= repeatsFactorSum_) {
        CopyIn(currentRepeatCount_, 0);
        CopyOutSplit(currentRepeatCount_);
    } else {
        int64_t loop = Ops::Base::CeilDiv(currentRepeatCount_, repeatsFactorSum_);
        int64_t tailLoopCount = currentRepeatCount_ - (loop - 1) * repeatsFactorSum_;
        for (int64_t i = 0; i < loop - 1; i++) {
            CopyIn(repeatsFactorSum_, i * repeatsFactorSum_);
            CopyOutSplit(repeatsFactorSum_);
        }
        CopyIn(tailLoopCount, (loop - 1) * repeatsFactorSum_);
        CopyOutSplit(tailLoopCount);
    }
    return;
}
} // namespace RepeatInterleave

#endif