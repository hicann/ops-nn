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
 * \file repeat_interleave.h
 * \brief
 */

#ifndef REPEAT_INTERLEAVE_REPEAT_H
#define REPEAT_INTERLEAVE_REPEAT_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "repeat_interleave_base.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint32_t UB_REPEAT_NUM_BUFFER = 16384;
constexpr uint32_t CP_AXIS = 2;
constexpr uint32_t REPEAT_AXIS = 1;

template <typename T>
__simd_vf__ inline void CopyOneCpToRepeatOutUnalignVf(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr,
                                                      uint16_t repeatTimes, uint16_t size, uint16_t stride,
                                                      int64_t oneCpSize)
{
    AscendC::Reg::UnalignRegForLoad uIn;
    AscendC::Reg::UnalignRegForStore uOut;
    AscendC::Reg::RegTensor<T> inputRegTensor;
    for (uint16_t j = 0; j < repeatTimes; j++) {
        __ubuf__ T* xOutCurPtr = xOutLocalPtr + j * oneCpSize;
        AscendC::Reg::LoadUnAlignPre(uIn, xInLocalPtr);
        for (uint16_t i = 0; i < size; i++) {
            AscendC::Reg::LoadUnAlign(inputRegTensor, uIn, xInLocalPtr + i * stride);
            AscendC::Reg::StoreUnAlign(xOutCurPtr, inputRegTensor, uOut, stride);
        }
        AscendC::Reg::StoreUnAlignPost(xOutCurPtr, uOut, 0);
    }
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
class RepeatInterleaveRepeatImpl {
public:
    __aicore__ inline RepeatInterleaveRepeatImpl(const RepeatInterleaveTilingKernelRepeat& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInX(Y_SIZE_T handleStartCpIdx, Y_SIZE_T blockCount, Y_SIZE_T dataCount);
    __aicore__ inline Y_SIZE_T GetCurRepeatNum(Y_SIZE_T curBatchIdx, Y_SIZE_T curRepeatIdx);
    __aicore__ inline void CopyOneCpToRepeatOut(const LocalTensor<X_T> xInLocal, LocalTensor<X_T> xOutLocal,
                                                uint16_t repeatNums, int64_t xInLocalOffset);
    __aicore__ inline void CopyXToMatchOut(Y_SIZE_T startCpIdx, Y_SIZE_T cpCount);
    __aicore__ inline void CopyMatchOutToY(LocalTensor<X_T> xOutLocal);
    __aicore__ inline void CopyOneSliceToOut(Y_SIZE_T yOffset, Y_SIZE_T curRepeatNum, Y_SIZE_T dataCount);
    __aicore__ inline void CopyOneSliceInX(Y_SIZE_T xOffset, Y_SIZE_T dataCount);
    __aicore__ inline Y_SIZE_T GetTotalProcessCp();
    __aicore__ inline void ProcessCpSplitMatchToUb();
    __aicore__ inline void ProcessSplitCpForKernel();
    __aicore__ inline void ProcessCpMatchToUb();
    __aicore__ inline void ProcessWholeCp();
    __aicore__ inline void GetRepeatIdxAndReverse(int64_t repeatsStart, int64_t repeatsEnd);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<REPEAT_T> repeatsGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<CAST_T> prefixSumGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xInQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> xInOutQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> xOutQueue_;
    TBuf<QuePosition::VECCALC> repeatBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    const RepeatInterleaveTilingKernelRepeat& tilingData_;

    Y_SIZE_T copyToMatchOutNum_{0};
    Y_SIZE_T copyToGmNum_{0};
    Y_SIZE_T outStartOffset_{0};
    Y_SIZE_T curRepeatHeadPos_{0};
    Y_SIZE_T startRepeatIdx_{0};
    Y_SIZE_T endRepeatIdx_{0};
    Y_SIZE_T startRepeatIdxRemain_{0};
    Y_SIZE_T endRepeatIdxRemain_{0};
    Y_SIZE_T repeatOffsetBase_{0};
    Y_SIZE_T curRepeatSize_{0};
    Y_SIZE_T endBatchIdx_{0};
    Y_SIZE_T startBatchIdx_{0};
};

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::Init(GM_ADDR x, GM_ADDR repeats,
                                                                                         GM_ADDR y, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    repeatsGm_.SetGlobalBuffer((__gm__ REPEAT_T*)repeats);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);

    pipe_.InitBuffer(repeatBuf_, UB_REPEAT_NUM_BUFFER); // 分配固定大小索引缓冲区
    pipe_.InitBuffer(tmpBuf_, platform::GetUbBlockSize());
    prefixSumGm_.SetGlobalBuffer((__gm__ CAST_T*)workspace);
    if (tilingData_.isSplitCPForKernel) {
        pipe_.InitBuffer(xInOutQueue_, DOUBLE_BUFFER,
                         ops::Aligned(static_cast<uint64_t>((tilingData_.eachCoreCpCount + 1) * sizeof(X_T)),
                                      static_cast<uint64_t>(platform::GetUbBlockSize())));
    } else if (tilingData_.isSplitCP) {
        pipe_.InitBuffer(xInOutQueue_, DOUBLE_BUFFER,
                         tilingData_.cpCountInUb * tilingData_.cpSliceFactorAlign * sizeof(X_T));
    } else {
        pipe_.InitBuffer(
            xInQueue_, DOUBLE_BUFFER,
            ops::Aligned(static_cast<uint64_t>(tilingData_.cpCountInUb * tilingData_.mergedDims[CP_AXIS] * sizeof(X_T)),
                         static_cast<uint64_t>(platform::GetUbBlockSize())));
        pipe_.InitBuffer(
            xOutQueue_, DOUBLE_BUFFER,
            ops::Aligned(static_cast<uint64_t>(tilingData_.cpCountInUb * tilingData_.mergedDims[CP_AXIS] * sizeof(X_T)),
                         static_cast<uint64_t>(platform::GetUbBlockSize())));
    }
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyMatchOutToY(
    LocalTensor<X_T> xOutLocal)
{
    xOutQueue_.EnQue(xOutLocal);
    xOutLocal = xOutQueue_.DeQue<X_T>();
    DataCopyExtParams outParams;
    outParams.blockCount = static_cast<uint16_t>(1);
    outParams.blockLen = static_cast<uint32_t>(copyToMatchOutNum_ * tilingData_.mergedDims[CP_AXIS]) * sizeof(X_T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outStartOffset_ + copyToGmNum_], xOutLocal, outParams);
    copyToGmNum_ += copyToMatchOutNum_ * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]);
    copyToMatchOutNum_ = 0;
    xOutQueue_.FreeTensor(xOutLocal);
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyOneCpToRepeatOut(
    const LocalTensor<X_T> xInLocal, LocalTensor<X_T> xOutLocal, uint16_t repeatNums, int64_t xInLocalOffset)
{
    __ubuf__ int8_t* xInLocalPtr = (__ubuf__ int8_t*)xInLocal.GetPhyAddr() +
                                   xInLocalOffset * tilingData_.mergedDims[CP_AXIS] * sizeof(X_T);
    __ubuf__ int8_t* xOutLocalPtr = (__ubuf__ int8_t*)xOutLocal.GetPhyAddr() +
                                    copyToMatchOutNum_ * tilingData_.mergedDims[CP_AXIS] * sizeof(X_T);
    uint32_t totalBytes = tilingData_.mergedDims[CP_AXIS] * sizeof(X_T);
    uint16_t stride = Ops::Base::GetVRegSize();
    uint16_t size = (totalBytes + stride - 1) / stride;
    CopyOneCpToRepeatOutUnalignVf<int8_t>(xInLocalPtr, xOutLocalPtr, repeatNums, size, stride,
                                          tilingData_.mergedDims[CP_AXIS] * sizeof(X_T));
    copyToMatchOutNum_ += repeatNums;
    return;
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline Y_SIZE_T RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::GetCurRepeatNum(
    Y_SIZE_T curBatchIdx, Y_SIZE_T curRepeatIdx)
{
    if (curBatchIdx == startBatchIdx_ && curRepeatIdx == startRepeatIdx_) {
        return startRepeatIdxRemain_;
    }
    if (curBatchIdx == endBatchIdx_ && curRepeatIdx == endRepeatIdx_) {
        return endRepeatIdxRemain_;
    }
    // UB中没有所需索引，需重新拷贝
    if (curRepeatIdx >= repeatOffsetBase_ + curRepeatSize_ || curRepeatIdx < repeatOffsetBase_) {
        Y_SIZE_T copyLen = 0;
        // blockFactorUbRepeat 不从 tiling获取，这里直接定义
        if (static_cast<Y_SIZE_T>(UB_REPEAT_NUM_BUFFER / sizeof(REPEAT_T)) >=
            static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS])) {
            // 拷贝全部索引
            if (endBatchIdx_ == curBatchIdx) {
                copyLen = endRepeatIdx_ - curRepeatIdx + 1;
                curRepeatHeadPos_ = curRepeatIdx;
            } else if ((endBatchIdx_ == curBatchIdx + 1) && (endRepeatIdx_ < curRepeatIdx)) {
                copyLen = static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) - curRepeatIdx;
                curRepeatHeadPos_ = curRepeatIdx;
            } else {
                copyLen = static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
                curRepeatHeadPos_ = 0;
            }
        } else {
            // 拷贝部分索引
            if (endBatchIdx_ == curBatchIdx) {
                copyLen = min((endRepeatIdx_ - curRepeatIdx + 1),
                              static_cast<Y_SIZE_T>(UB_REPEAT_NUM_BUFFER / sizeof(REPEAT_T)));
                curRepeatHeadPos_ = curRepeatIdx;
            } else {
                copyLen = min(static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) - curRepeatIdx,
                              static_cast<Y_SIZE_T>(UB_REPEAT_NUM_BUFFER / sizeof(REPEAT_T)));
                curRepeatHeadPos_ = curRepeatIdx;
            }
        }

        curRepeatSize_ = copyLen;
        repeatOffsetBase_ = curRepeatHeadPos_;
        DataCopyExtParams inParams = {1, static_cast<uint32_t>(copyLen * sizeof(REPEAT_T)), 0, 0, 0};
        DataCopyPadExtParams<REPEAT_T> padParams = {false, 0, 0, 0};
        LocalTensor<REPEAT_T> tempLocal = repeatBuf_.Get<REPEAT_T>();
        DataCopyPad(tempLocal, repeatsGm_[curRepeatHeadPos_], inParams, padParams);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<REPEAT_T> repeatLocal = repeatBuf_.Get<REPEAT_T>();
    return static_cast<Y_SIZE_T>(repeatLocal.GetValue(curRepeatIdx - curRepeatHeadPos_));
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyXToMatchOut(Y_SIZE_T startCpIdx,
                                                                                                    Y_SIZE_T cpCount)
{
    LocalTensor<X_T> xInLocal = xInQueue_.DeQue<X_T>();
    LocalTensor<X_T> xOutLocal = xOutQueue_.AllocTensor<X_T>();

    for (Y_SIZE_T i = 0; i < cpCount; i++) {
        Y_SIZE_T curCpIdx = i + startCpIdx;
        Y_SIZE_T curBatchIdx = curCpIdx / static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
        Y_SIZE_T curRepeatIdx = curCpIdx % static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
        Y_SIZE_T curRepeatNum = GetCurRepeatNum(curBatchIdx, curRepeatIdx);
        if (curRepeatNum == 0) {
            continue;
        }
        Y_SIZE_T loopSize = (curRepeatNum + static_cast<Y_SIZE_T>(tilingData_.cpCountInUb) - 1) /
                            static_cast<Y_SIZE_T>(tilingData_.cpCountInUb);
        Y_SIZE_T mainRepeatNums = static_cast<Y_SIZE_T>(tilingData_.cpCountInUb);
        Y_SIZE_T tailRepeatNums = curRepeatNum - static_cast<Y_SIZE_T>(tilingData_.cpCountInUb) * (loopSize - 1);
        for (Y_SIZE_T loopIdx = 0; loopIdx < (loopSize - 1); loopIdx++) {
            if ((mainRepeatNums + copyToMatchOutNum_) > static_cast<Y_SIZE_T>(tilingData_.cpCountInUb)) {
                CopyMatchOutToY(xOutLocal);
                xOutLocal = xOutQueue_.AllocTensor<X_T>();
            }
            CopyOneCpToRepeatOut(xInLocal, xOutLocal, mainRepeatNums, i);
        }
        if ((tailRepeatNums + copyToMatchOutNum_) > static_cast<Y_SIZE_T>(tilingData_.cpCountInUb)) {
            CopyMatchOutToY(xOutLocal);
            xOutLocal = xOutQueue_.AllocTensor<X_T>();
        }
        CopyOneCpToRepeatOut(xInLocal, xOutLocal, tailRepeatNums, i);
    }
    // xOutLocal 这里一定是复制的，所以最后还需要再搬出依次就处理完了。
    CopyMatchOutToY(xOutLocal);
    xInQueue_.FreeTensor(xInLocal);
    return;
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyOneSliceToOut(
    Y_SIZE_T yOffset, Y_SIZE_T curRepeatNum, Y_SIZE_T dataCount)
{
    LocalTensor<X_T> xInLocal = xInOutQueue_.DeQue<X_T>();
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = static_cast<uint32_t>(dataCount) * sizeof(X_T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    for (Y_SIZE_T yOffIdx = 0; yOffIdx < curRepeatNum; yOffIdx++) {
        DataCopyPad(yGm_[yOffset + yOffIdx * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS])], xInLocal,
                    outParams);
    }
    xInOutQueue_.FreeTensor(xInLocal);
    return;
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyOneSliceInX(Y_SIZE_T xOffset,
                                                                                                    Y_SIZE_T dataCount)
{
    Y_SIZE_T dataLen = dataCount * sizeof(X_T);

    DataCopyExtParams inParams = {1, static_cast<uint32_t>(dataLen), 0, 0, 0};
    DataCopyPadExtParams<X_T> padParams = {false, 0, 0, 0};
    LocalTensor<X_T> xInLocal = xInOutQueue_.AllocTensor<X_T>();
    DataCopyPad(xInLocal, xGm_[xOffset], inParams, padParams);
    xInOutQueue_.EnQue(xInLocal);
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::CopyInX(Y_SIZE_T handleStartCpIdx,
                                                                                            Y_SIZE_T blockCount,
                                                                                            Y_SIZE_T dataCount)
{
    /* 源数据的偏移 */
    Y_SIZE_T xOffset = handleStartCpIdx * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]);
    Y_SIZE_T dataLen = dataCount * sizeof(X_T);

    DataCopyExtParams inParams = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen), 0, 0, 0};
    DataCopyPadExtParams<X_T> padParams = {false, 0, 0, 0};
    LocalTensor<X_T> xInLocal = xInQueue_.AllocTensor<X_T>();
    DataCopyPad<X_T, PaddingMode::Compact>(xInLocal, xGm_[xOffset], inParams, padParams);
    xInQueue_.EnQue(xInLocal);
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline Y_SIZE_T RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::GetTotalProcessCp()
{
    Y_SIZE_T totalProCp = static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) - startRepeatIdx_ + endRepeatIdx_ +
                          1;
    if (startBatchIdx_ == endBatchIdx_) {
        totalProCp = endRepeatIdx_ - startRepeatIdx_ + 1;
    } else if (endBatchIdx_ > startBatchIdx_ + 1) {
        totalProCp += (endBatchIdx_ - 1 - startBatchIdx_) * static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
    }
    return totalProCp;
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::ProcessCpMatchToUb()
{
    Y_SIZE_T totalProCp = GetTotalProcessCp();
    Y_SIZE_T loopSize = (totalProCp + static_cast<Y_SIZE_T>(tilingData_.cpCountInUb) - 1) /
                        static_cast<Y_SIZE_T>(tilingData_.cpCountInUb);
    Y_SIZE_T mainCpNum = static_cast<Y_SIZE_T>(tilingData_.cpCountInUb);
    Y_SIZE_T tailCpNum = totalProCp - static_cast<Y_SIZE_T>(tilingData_.cpCountInUb) * (loopSize - 1);
    Y_SIZE_T handleStartCpIdx = 0;
    Y_SIZE_T processCpNum = mainCpNum;
    for (Y_SIZE_T loopIdx = 0; loopIdx < loopSize; loopIdx++) {
        handleStartCpIdx = startBatchIdx_ * static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) +
                           startRepeatIdx_ + loopIdx * mainCpNum;
        if (loopIdx == loopSize - 1) {
            processCpNum = tailCpNum;
        }
        CopyInX(handleStartCpIdx, processCpNum, static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]));
        CopyXToMatchOut(handleStartCpIdx, processCpNum);
    }
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::ProcessCpSplitMatchToUb()
{
    Y_SIZE_T totalProCp = GetTotalProcessCp();
    Y_SIZE_T handleStartCpIdx = 0;
    for (Y_SIZE_T loopIdx = 0; loopIdx < totalProCp; loopIdx++) {
        handleStartCpIdx = startBatchIdx_ * static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) +
                           startRepeatIdx_ + loopIdx;
        Y_SIZE_T curBatchIdx = handleStartCpIdx / static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
        Y_SIZE_T curRepeatIdx = handleStartCpIdx % static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]);
        Y_SIZE_T curRepeatNum = GetCurRepeatNum(curBatchIdx, curRepeatIdx);
        for (Y_SIZE_T cpSliceIdx = 0; cpSliceIdx < static_cast<Y_SIZE_T>(tilingData_.cpSliceNum); cpSliceIdx++) {
            Y_SIZE_T cpSliceFactor = static_cast<Y_SIZE_T>(tilingData_.cpSliceFactorAlign);
            if (cpSliceIdx == tilingData_.cpSliceNum - 1) {
                cpSliceFactor = static_cast<Y_SIZE_T>(tilingData_.cpSliceFactorTail);
            }
            Y_SIZE_T xOffset = handleStartCpIdx * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]) +
                               cpSliceIdx * static_cast<Y_SIZE_T>(tilingData_.cpSliceFactorAlign);
            CopyOneSliceInX(xOffset, cpSliceFactor);
            Y_SIZE_T yOffset = outStartOffset_ + copyToGmNum_ +
                               cpSliceIdx * static_cast<Y_SIZE_T>(tilingData_.cpSliceFactorAlign);
            CopyOneSliceToOut(yOffset, curRepeatNum, cpSliceFactor);
        }
        copyToGmNum_ += curRepeatNum * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]);
    }
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::ProcessWholeCp()
{
    Y_SIZE_T curCoreCpCount = 0;
    Y_SIZE_T startCpCount = 0;
    Y_SIZE_T endCpCount = 0;
    if (static_cast<Y_SIZE_T>(GetBlockIdx()) < static_cast<Y_SIZE_T>(tilingData_.tailCoreCpCount)) {
        curCoreCpCount = static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) + 1;
        startCpCount = static_cast<Y_SIZE_T>(GetBlockIdx()) * (static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) + 1);
    } else {
        curCoreCpCount = tilingData_.eachCoreCpCount;
        startCpCount = static_cast<Y_SIZE_T>(GetBlockIdx()) * static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) +
                       static_cast<Y_SIZE_T>(tilingData_.tailCoreCpCount);
    }
    endCpCount = startCpCount + curCoreCpCount - 1;
    outStartOffset_ = startCpCount * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]);

    startBatchIdx_ = startCpCount / static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    Y_SIZE_T startRepeatNumIdx = startCpCount % static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    endBatchIdx_ = endCpCount / static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    Y_SIZE_T endRepeatNumIdx = endCpCount % static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    GetRepeatIdxAndReverse(startRepeatNumIdx, endRepeatNumIdx);
    if (tilingData_.isSplitCP) {
        ProcessCpSplitMatchToUb();
        return;
    }
    ProcessCpMatchToUb();
    return;
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::GetRepeatIdxAndReverse(
    int64_t repeatsStart, int64_t repeatsEnd)
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    int64_t repeatsNum = repeatsEnd - repeatsStart + 1;
    // 如果跨batch，则取最大值，保证一定取到正确的未处理repeatNum
    if (startBatchIdx_ < endBatchIdx_) {
        repeatsNum = tilingData_.totalRepeatSum;
    }

    LocalTensor<CAST_T> tmpLocal = tmpBuf_.Get<CAST_T>();

    asc_vf_call<SimtSearchStartEnd<REPEAT_T, CAST_T, CAST_T>>(
        dim3(SEARCH_THREAD_NUM), tilingData_.mergedDims[REPEAT_AXIS], repeatsNum, repeatsStart, repeatsEnd,
        (__ubuf__ CAST_T*)(tmpLocal.GetPhyAddr()), (__gm__ CAST_T*)(prefixSumGm_.GetPhyAddr()));

    auto sWiatVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(sWiatVEventID);
    WaitFlag<HardEvent::V_S>(sWiatVEventID);

    startRepeatIdx_ = tmpLocal.GetValue(0);
    startRepeatIdxRemain_ = tmpLocal.GetValue(1);
    endRepeatIdx_ = tmpLocal.GetValue(2);
    endRepeatIdxRemain_ = tmpLocal.GetValue(3);
}

// 此场景下，每个核只会处理单行数据，且一定会单次搬运完毕
template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::ProcessSplitCpForKernel()
{
    Y_SIZE_T blockRow = static_cast<Y_SIZE_T>(GetBlockIdx()) / static_cast<Y_SIZE_T>(tilingData_.cpSplitBlocks);
    Y_SIZE_T blockCol = static_cast<Y_SIZE_T>(GetBlockIdx()) % static_cast<Y_SIZE_T>(tilingData_.cpSplitBlocks);
    startBatchIdx_ = blockRow / static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    Y_SIZE_T startRepeatNumIdx = blockRow % static_cast<Y_SIZE_T>(tilingData_.totalRepeatSum);
    Y_SIZE_T curCoreCpCount = 0;
    Y_SIZE_T cpOffset = 0;
    if (blockCol < static_cast<Y_SIZE_T>(tilingData_.tailCoreCpCount)) {
        curCoreCpCount = static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) + 1;
        cpOffset = blockCol * (static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) + 1);
    } else {
        cpOffset = blockCol * static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount) +
                   static_cast<Y_SIZE_T>(tilingData_.tailCoreCpCount);
        curCoreCpCount = static_cast<Y_SIZE_T>(tilingData_.eachCoreCpCount);
    }
    outStartOffset_ = blockRow * static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]) + cpOffset;
    endBatchIdx_ = startBatchIdx_;
    GetRepeatIdxAndReverse(startRepeatNumIdx, startRepeatNumIdx);
    endRepeatIdx_ = startRepeatIdx_;
    endRepeatIdxRemain_ = startRepeatIdxRemain_;
    Y_SIZE_T xOffset = (startBatchIdx_ * static_cast<Y_SIZE_T>(tilingData_.mergedDims[REPEAT_AXIS]) + startRepeatIdx_) *
                           static_cast<Y_SIZE_T>(tilingData_.mergedDims[CP_AXIS]) +
                       cpOffset;
    CopyOneSliceInX(xOffset, curCoreCpCount);
    Y_SIZE_T curRepeatNum = GetCurRepeatNum(startBatchIdx_, startRepeatIdx_);
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    CopyOneSliceToOut(outStartOffset_, curRepeatNum, curCoreCpCount);
}

template <typename X_T, typename Y_SIZE_T, typename REPEAT_T, typename CAST_T>
__aicore__ inline void RepeatInterleaveRepeatImpl<X_T, Y_SIZE_T, REPEAT_T, CAST_T>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    if (tilingData_.isSplitCPForKernel) {
        ProcessSplitCpForKernel();
    } else {
        ProcessWholeCp();
    }
}
} // namespace RepeatInterleave

#endif // REPEAT_INTERLEAVE_REPEAT_H
