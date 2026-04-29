/**
 • Copyright (c) 2026 Huawei Technologies Co., Ltd.
 • This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 • CANN Open Software License Agreement Version 2.0 (the "License")
 • Please refer to the License for details. You may not use this file except in compliance with the License.
 • THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 • INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 • See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SCATTER_ADD_WITH_SORTED_SIMD_DETERM_H
#define SCATTER_ADD_WITH_SORTED_SIMD_DETERM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"
#include "scatter_add_with_sorted_simd_common.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

template <typename T, typename U, bool withPos>
class ScatterAddWithSortedSimdDterm {
public:
    __aicore__ inline ScatterAddWithSortedSimdDterm(void)
    {}

    __aicore__ inline void CopyIn(
        LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);
    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
        const ScatterAddWithSortedSimdTilingData* tilingData);
    __aicore__ inline void CopyOut(
        GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);
    __aicore__ inline void CopyInIndices(
        LocalTensor<U> dstLocal, GlobalTensor<U> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);
    __aicore__ inline void CopyOutIndexToWorkspace(LocalTensor<U>& tmpLocal);
    __aicore__ inline void CopyOutToWorkspace(
        LocalTensor<T>& dataLocal, int32_t burstLen, int64_t colOffset, int32_t writePosition);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeSumAndCopyOut(
        LocalTensor<T>& yLocal, int32_t curLoopRows, int32_t curLoopCols, int32_t curLoopColsAlign, int64_t colOffset,
        U& curId, int64_t row, int64_t indicesGmOffset);

private:
    GlobalTensor<T> varGm_;
    GlobalTensor<U> indicesGm_;
    GlobalTensor<U> posGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> varRefGm_;
    GlobalTensor<T> sumWorkspace_;
    GlobalTensor<U> indexWorkspace_;

    TQue<QuePosition::VECIN, kBufferNum> updatesQueue_;
    TQue<QuePosition::VECIN, kBufferNum> indicesQueue_;
    TQue<QuePosition::VECIN, kBufferNum> posQueue_;
    TQue<QuePosition::VECIN, 1> lastCoreLastIndexQueue_;
    TQue<QuePosition::VECIN, 1> nextCoreFirstIndexQueue_;
    TQue<QuePosition::VECOUT, TMP_BUFFER_NUM> tmpQue_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    const ScatterAddWithSortedSimdTilingData* tilingData_ = nullptr;

    uint32_t rowCoreIdx_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t colCoreIdx_ = 0;
    int64_t rowGmOffset_ = 0;
    int64_t rowUbLoop_ = 0;
    int64_t colGmOffset_ = 0;
    int64_t colUbLoop_ = 0;
    int64_t tailLoopCols_ = 0;
    int64_t tailLoopRows_ = 0;
    int64_t normalLoopRows_ = 0;
    int64_t normalLoopCols_ = 0;

    U preId_ = static_cast<U>(-1);
    U prevCoreLastIndex_ = static_cast<U>(-2);
    U nextCoreFirstIndex_ = static_cast<U>(-2);

    U headIndex_ = static_cast<U>(-1);
    U tailIndex_ = static_cast<U>(-1);

    bool isStartRowCore_ = false;
    bool isEndRowCore_ = false;

    bool headToWorkspace_ = false;
    bool tailToWorkspace_ = false;

    constexpr static int32_t blockNumT_ = platform::GetUbBlockSize() / sizeof(T);
};

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::Init(
    GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
    const ScatterAddWithSortedSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (!KernelUtil::InitSimdBase(
            tilingData_, blockIdx_, rowCoreIdx_, colCoreIdx_, isStartRowCore_, isEndRowCore_, rowGmOffset_,
            colGmOffset_, rowUbLoop_, colUbLoop_, normalLoopRows_, tailLoopRows_, normalLoopCols_, tailLoopCols_,
            updatesGm_, indicesGm_, updates, indices)) {
        return;
    }

    int64_t sumAreaBytes = static_cast<int64_t>(tilingData_->coreNumInRow) * kDouble * tilingData_->vecAlignSize;
    int64_t indexAreaOffsetBytes = sumAreaBytes + static_cast<int64_t>(rowCoreIdx_) * kCacheLineSize;

    int64_t sumOffsetT = static_cast<int64_t>(rowCoreIdx_) * kDouble * tilingData_->vecAlignSize / sizeof(T);
    int64_t indexOffsetU = (indexAreaOffsetBytes + sizeof(U) - 1) / sizeof(U);

    if constexpr (withPos) {
        posGm_.SetGlobalBuffer((__gm__ U*)(pos));
    }
    varRefGm_.SetGlobalBuffer((__gm__ T*)(varRef));

    sumWorkspace_.SetGlobalBuffer((__gm__ T*)workspace + sumOffsetT);
    indexWorkspace_.SetGlobalBuffer((__gm__ U*)workspace + indexOffsetU);

    pipeIn.InitBuffer(updatesQueue_, kBufferNum, tilingData_->updatesBufferSize);
    pipeIn.InitBuffer(indicesQueue_, kBufferNum, tilingData_->indicesBufferSize);
    if constexpr (withPos) {
        pipeIn.InitBuffer(posQueue_, kBufferNum, tilingData_->posBufferSize);
    }
    pipeIn.InitBuffer(lastCoreLastIndexQueue_, 1, tilingData_->FrontAndBackIndexSize / 2);
    pipeIn.InitBuffer(nextCoreFirstIndexQueue_, 1, tilingData_->FrontAndBackIndexSize / 2);

    pipeIn.InitBuffer(yBuf_, tilingData_->outBufferSize);
    pipeIn.InitBuffer(tmpQue_, TMP_BUFFER_NUM, tilingData_->outBufferSize);
    pipeIn.InitBuffer(tmpBuf_, platform::GetUbBlockSize());
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyIn<T>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::CopyInIndices(
    LocalTensor<U> dstLocal, GlobalTensor<U> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyInIndices<U>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyOut<T>(dstGm, srcLocal, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::CopyOutToWorkspace(
    LocalTensor<T>& dataLocal, int32_t burstLen, int64_t colOffset, int32_t writePosition)
{
    DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = static_cast<uint32_t>(burstLen) * sizeof(T);
    ext.srcStride = 0;
    ext.dstStride = 0;

    uint64_t workspaceOffset = static_cast<uint64_t>(writePosition) * (tilingData_->vecAlignSize / sizeof(T)) +
                               static_cast<uint64_t>(colOffset);
    DataCopyPad(sumWorkspace_[workspaceOffset], dataLocal, ext);
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::CopyOutIndexToWorkspace(LocalTensor<U>& tmpLocal)
{
    DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = kDouble * sizeof(U);
    ext.srcStride = 0;
    ext.dstStride = 0;
    DataCopyPad(indexWorkspace_, tmpLocal, ext);
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::ComputeSumAndCopyOut(
    LocalTensor<T>& yLocal, int32_t curLoopRows, int32_t curLoopCols, int32_t curLoopColsAlign, int64_t colOffset,
    U& curId, int64_t row, int64_t indicesGmOffset)
{
    LocalTensor<U> indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<U> posLocal;
    if constexpr (withPos) {
        posLocal = posQueue_.DeQue<U>();
    }
    event_t evMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evMte2S);
    WaitFlag<HardEvent::MTE2_S>(evMte2S);

    if (row == 0) {
        headIndex_ = indicesLocal.GetValue(0);
        if (!isStartRowCore_ && headIndex_ == prevCoreLastIndex_) {
            headToWorkspace_ = true;
        }
    }

    if (row == rowUbLoop_ - 1) {
        tailIndex_ = indicesLocal.GetValue(curLoopRows - 1);
        if (!isEndRowCore_ && tailIndex_ == nextCoreFirstIndex_) {
            tailToWorkspace_ = true;
        }
    }

    for (int32_t i = 0; i < curLoopRows; i++) {
        curId = indicesLocal.GetValue(i);
        uint64_t updatesOffset = 0;
        if constexpr (withPos) {
            U posId = posLocal.GetValue(i);
            updatesOffset = static_cast<uint64_t>(posId) * tilingData_->updatesInner + colOffset;
        } else {
            updatesOffset = static_cast<uint64_t>(indicesGmOffset + i) * tilingData_->updatesInner + colOffset;
        }
        LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
        CopyIn(updatesLocal, updatesGm_, updatesOffset, 1, static_cast<uint32_t>(curLoopCols));
        updatesQueue_.EnQue(updatesLocal);
        updatesLocal = updatesQueue_.DeQue<T>();

        if (KernelUtil::AccumulateOrInit(yLocal, updatesLocal, preId_, curId, curLoopCols, curLoopColsAlign)) {
            LocalTensor<T> tmpLocal = tmpQue_.AllocTensor<T>();
            DataCopy(tmpLocal, yLocal, curLoopColsAlign);
            tmpQue_.EnQue(tmpLocal);
            LocalTensor<T> outLocal = tmpQue_.DeQue<T>();

            if (preId_ == headIndex_ && headToWorkspace_) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(outLocal, curLoopCols, colOffset, 0);
            } else if (preId_ == tailIndex_ && tailToWorkspace_) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(outLocal, curLoopCols, colOffset, 1);
            } else {
                uint64_t dstOffset = static_cast<uint64_t>(preId_) * tilingData_->updatesInner + colOffset;
                KernelUtil::WaitVToMte3();
                SetAtomicAdd<T>();
                CopyOut(varRefGm_, outLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
                SetAtomicNone();
            }

            tmpQue_.FreeTensor(outLocal);
            preId_ = curId;
            DataCopy(yLocal, updatesLocal, curLoopColsAlign);
        }

        updatesQueue_.FreeTensor(updatesLocal);
    }

    indicesQueue_.FreeTensor(indicesLocal);
    if constexpr (withPos) {
        posQueue_.FreeTensor(posLocal);
    }
}

template <typename T, typename U, bool withPos>
__aicore__ inline void ScatterAddWithSortedSimdDterm<T, U, withPos>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    if (!isStartRowCore_) {
        LocalTensor<U> tmp = lastCoreLastIndexQueue_.AllocTensor<U>();
        CopyInIndices(tmp, indicesGm_, static_cast<uint64_t>(rowCoreIdx_) * tilingData_->normalCoreRowNum - 1, 1, 1);

        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ev);
        WaitFlag<HardEvent::MTE2_S>(ev);

        prevCoreLastIndex_ = tmp.GetValue(0);
        lastCoreLastIndexQueue_.FreeTensor(tmp);
    } else {
        prevCoreLastIndex_ = static_cast<U>(-1);
    }

    if (!isEndRowCore_) {
        LocalTensor<U> tmp = nextCoreFirstIndexQueue_.AllocTensor<U>();
        CopyInIndices(tmp, indicesGm_, static_cast<uint64_t>(rowCoreIdx_ + 1) * tilingData_->normalCoreRowNum, 1, 1);

        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ev);
        WaitFlag<HardEvent::MTE2_S>(ev);

        nextCoreFirstIndex_ = tmp.GetValue(0);
        nextCoreFirstIndexQueue_.FreeTensor(tmp);
    } else {
        nextCoreFirstIndex_ = static_cast<U>(-1);
    }

    LocalTensor<T> yLocal = yBuf_.Get<T>();

    for (int64_t col = 0; col < colUbLoop_; col++) {
        preId_ = static_cast<U>(-1);
        U curId = static_cast<U>(-1);

        headIndex_ = static_cast<U>(-1);
        tailIndex_ = static_cast<U>(-1);
        headToWorkspace_ = false;
        tailToWorkspace_ = false;

        int64_t curLoopCols = (col == colUbLoop_ - 1) ? tailLoopCols_ : normalLoopCols_;
        int64_t colOffset = colGmOffset_ + col * normalLoopCols_;
        int64_t curLoopColsAlign = (curLoopCols + blockNumT_ - 1) / blockNumT_ * blockNumT_;

        for (int64_t row = 0; row < rowUbLoop_; row++) {
            int64_t indicesGmOffset = rowGmOffset_ + row * normalLoopRows_;
            int64_t curLoopRows = (row == rowUbLoop_ - 1) ? tailLoopRows_ : normalLoopRows_;
            KernelUtil::CopyInRowIndices<U, withPos>(
                indicesQueue_, posQueue_, indicesGm_, posGm_, indicesGmOffset, static_cast<uint32_t>(curLoopRows));
            ComputeSumAndCopyOut(
                yLocal, static_cast<int32_t>(curLoopRows), static_cast<int32_t>(curLoopCols),
                static_cast<int32_t>(curLoopColsAlign), colOffset, curId, row, indicesGmOffset);
        }
        if (preId_ != static_cast<U>(-1)) {
            bool isHeadBoundary = (preId_ == headIndex_ && headToWorkspace_);
            bool isTailBoundary = (preId_ == tailIndex_ && tailToWorkspace_);

            LocalTensor<T> tmpLocal = tmpQue_.AllocTensor<T>();
            DataCopy(tmpLocal, yLocal, curLoopColsAlign);
            tmpQue_.EnQue(tmpLocal);
            LocalTensor<T> outLocal = tmpQue_.DeQue<T>();

            if (isHeadBoundary && isTailBoundary) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(outLocal, curLoopCols, colOffset, 1);
            } else if (isTailBoundary) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(outLocal, curLoopCols, colOffset, 1);
            } else if (isHeadBoundary) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(outLocal, curLoopCols, colOffset, 0);
            } else {
                uint64_t dstOffset = static_cast<uint64_t>(preId_) * tilingData_->updatesInner + colOffset;
                KernelUtil::WaitVToMte3();
                SetAtomicAdd<T>();
                CopyOut(varRefGm_, outLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
                SetAtomicNone();
            }
            tmpQue_.FreeTensor(outLocal);
        }
    }
    LocalTensor<U> tmpIndicesLocal = tmpBuf_.Get<U>();
    U headFinalIndex = headToWorkspace_ ? headIndex_ : static_cast<U>(-1);
    U tailFinalIndex = tailToWorkspace_ ? tailIndex_ : static_cast<U>(-1);

    if (headToWorkspace_ && tailToWorkspace_ && headIndex_ == tailIndex_) {
        headFinalIndex = static_cast<U>(-1);
    }

    tmpIndicesLocal.SetValue(0, headFinalIndex);
    tmpIndicesLocal.SetValue(1, tailFinalIndex);

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(ev);
    WaitFlag<HardEvent::S_MTE3>(ev);

    CopyOutIndexToWorkspace(tmpIndicesLocal);
}

} // namespace ScatterAddWithSorted
#endif // SCATTER_ADD_WITH_SORTED_SIMD_DETERM_H