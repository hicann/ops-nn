/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ADD_WITH_SORTED_SIMD_H
#define SCATTER_ADD_WITH_SORTED_SIMD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"
#include "scatter_add_with_sorted_simd_common.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

template <typename T, typename U, bool updatesIsScalar, bool withPos>
class ScatterAddWithSortedSIMD {
public:
    __aicore__ inline ScatterAddWithSortedSIMD(void)
    {}
    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
        const ScatterAddWithSortedSimdTilingData* tilingData);

    __aicore__ inline void CopyIn(
        LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void CopyInIndices(
        LocalTensor<U> dstLocal, GlobalTensor<U> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void CopyOut(
        GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void BroadcastUpdatesScalar(
        LocalTensor<T> updatesLocal, GlobalTensor<T> updatesGm, int32_t count);

    __aicore__ inline void ComputeSumAndCopyOut(
        LocalTensor<T>& yLocal, int32_t curLoopRows, int32_t curLoopCols, int32_t curLoopColsAlign, int64_t colOffset,
        U& curId, int64_t indicesGmOffset);

    __aicore__ inline void Process();

private:
    GlobalTensor<T> varGm_;
    GlobalTensor<U> indicesGm_;
    GlobalTensor<U> posGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> varRefGm_;

    TQue<QuePosition::VECIN, kBufferNum> updatesQueue_;
    TQue<QuePosition::VECIN, kBufferNum> indicesQueue_;
    TQue<QuePosition::VECIN, kBufferNum> posQueue_;
    TQue<QuePosition::VECOUT, TMP_BUFFER_NUM> tmpQue_;
    TBuf<QuePosition::VECCALC> yBuf_;

    const ScatterAddWithSortedSimdTilingData* tilingData_ = nullptr;

    uint32_t blockIdx_ = 0;
    uint32_t rowCoreIdx_ = 0;
    uint32_t colCoreIdx_ = 0;

    int64_t rowGmOffset_ = 0;
    int64_t colGmOffset_ = 0;

    int64_t rowUbLoop_ = 0;
    int64_t colUbLoop_ = 0;

    int64_t normalLoopRows_ = 0;
    int64_t tailLoopRows_ = 0;
    int64_t normalLoopCols_ = 0;
    int64_t tailLoopCols_ = 0;

    U preId_ = static_cast<U>(-1);

    bool isStartRowCore_ = false;
    bool isEndRowCore_ = false;

    constexpr static int32_t blockNumT_ = platform::GetUbBlockSize() / sizeof(T);
};

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::Init(
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

    if constexpr (withPos) {
        posGm_.SetGlobalBuffer((__gm__ U*)(pos));
    }
    varRefGm_.SetGlobalBuffer((__gm__ T*)(varRef));

    pipeIn.InitBuffer(updatesQueue_, kBufferNum, tilingData_->updatesBufferSize);
    pipeIn.InitBuffer(indicesQueue_, kBufferNum, tilingData_->indicesBufferSize);
    if constexpr (withPos) {
        pipeIn.InitBuffer(posQueue_, kBufferNum, tilingData_->posBufferSize);
    }
    pipeIn.InitBuffer(yBuf_, tilingData_->outBufferSize);
    pipeIn.InitBuffer(tmpQue_, TMP_BUFFER_NUM, tilingData_->outBufferSize);
}

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyIn<T>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::CopyInIndices(
    LocalTensor<U> dstLocal, GlobalTensor<U> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyInIndices<U>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyOut<T>(dstGm, srcLocal, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::BroadcastUpdatesScalar(
    LocalTensor<T> updatesLocal, GlobalTensor<T> updatesGm, int32_t count)
{
    KernelUtil::BroadcastScalar<T>(updatesLocal, updatesGm, static_cast<uint32_t>(count));
}

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::ComputeSumAndCopyOut(
    LocalTensor<T>& yLocal, int32_t curLoopRows, int32_t curLoopCols, int32_t curLoopColsAlign, int64_t colOffset,
    U& curId, int64_t indicesGmOffset)
{
    LocalTensor<U> indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<U> posLocal;
    if constexpr (withPos) {
        posLocal = posQueue_.DeQue<U>();
    }
    event_t evMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evMte2S);
    WaitFlag<HardEvent::MTE2_S>(evMte2S);

    for (int32_t i = 0; i < curLoopRows; i++) {
        curId = indicesLocal.GetValue(i);
        if constexpr (updatesIsScalar) {
            LocalTensor<T> upd = updatesQueue_.AllocTensor<T>();
            BroadcastUpdatesScalar(upd, updatesGm_, curLoopColsAlign);
            updatesQueue_.EnQue(upd);
        } else {
            LocalTensor<T> upd = updatesQueue_.AllocTensor<T>();
            uint64_t updatesOffset = 0;
            if constexpr (withPos) {
                U posId = posLocal.GetValue(i);
                updatesOffset = static_cast<uint64_t>(posId) * tilingData_->updatesInner + colOffset;
            } else {
                updatesOffset = static_cast<uint64_t>(indicesGmOffset + i) * tilingData_->updatesInner + colOffset;
            }
            CopyIn(upd, updatesGm_, updatesOffset, 1, static_cast<uint32_t>(curLoopCols));
            updatesQueue_.EnQue(upd);
        }

        LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();

        if (KernelUtil::AccumulateOrInit(yLocal, updatesLocal, preId_, curId, curLoopCols, curLoopColsAlign)) {
            LocalTensor<T> tmpLocal = tmpQue_.AllocTensor<T>();
            DataCopy(tmpLocal, yLocal, curLoopColsAlign);
            tmpQue_.EnQue(tmpLocal);
            LocalTensor<T> outLocal = tmpQue_.DeQue<T>();

            uint64_t dstOffset = static_cast<uint64_t>(preId_) * tilingData_->updatesInner + colOffset;
            KernelUtil::WaitVToMte3();
            SetAtomicAdd<T>();
            CopyOut(varRefGm_, outLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
            SetAtomicNone();

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

template <typename T, typename U, bool updatesIsScalar, bool withPos>
__aicore__ inline void ScatterAddWithSortedSIMD<T, U, updatesIsScalar, withPos>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }
    LocalTensor<T> yLocal = yBuf_.Get<T>();
    for (int64_t col = 0; col < colUbLoop_; col++) {
        preId_ = static_cast<U>(-1);
        U curId = static_cast<U>(-1);
        int64_t curLoopCols = (col == colUbLoop_ - 1) ? tailLoopCols_ : normalLoopCols_;
        int64_t curLoopColsAlign = (curLoopCols + blockNumT_ - 1) / blockNumT_ * blockNumT_;
        int64_t colOffset = colGmOffset_ + col * normalLoopCols_;

        for (int64_t row = 0; row < rowUbLoop_; row++) {
            int64_t curLoopRows = (row == rowUbLoop_ - 1) ? tailLoopRows_ : normalLoopRows_;
            int64_t indicesGmOffset = rowGmOffset_ + row * normalLoopRows_;

            KernelUtil::CopyInRowIndices<U, withPos>(
                indicesQueue_, posQueue_, indicesGm_, posGm_, indicesGmOffset, static_cast<uint32_t>(curLoopRows));
            ComputeSumAndCopyOut(
                yLocal, static_cast<int32_t>(curLoopRows), static_cast<int32_t>(curLoopCols),
                static_cast<int32_t>(curLoopColsAlign), colOffset, curId, indicesGmOffset);
        }
        if (preId_ != static_cast<U>(-1)) {
            KernelUtil::WaitVToMte3();
            SetAtomicAdd<T>();
            uint64_t dstOffset = static_cast<uint64_t>(preId_) * tilingData_->updatesInner + colOffset;
            CopyOut(varRefGm_, yLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
            SetAtomicNone();
            KernelUtil::WaitMte3ToV();
        }
    }
}

} // namespace ScatterAddWithSorted
#endif // SCATTER_ADD_WITH_SORTED_SIMD_H