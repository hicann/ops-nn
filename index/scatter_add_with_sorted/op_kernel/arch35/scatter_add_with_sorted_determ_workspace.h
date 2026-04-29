/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_H
#define SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"
#include "scatter_add_with_sorted_simd_common.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

template <typename T, typename U>
class ScatterAddWithSortedSimdDtermWorkspace {
public:
    __aicore__ inline ScatterAddWithSortedSimdDtermWorkspace(void)
    {}

    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
        const ScatterAddWithSortedSimdTilingData* tilingData);

    __aicore__ inline void CopyInIndicesWorkspace(LocalTensor<U>& indicesWorkspaceLocal);

    __aicore__ inline void CopyIn(
        LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void CopyOut(
        GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
        uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void Process();

private:
    GlobalTensor<U> indexWorkspace_;
    GlobalTensor<T> sumWorkspace_;
    GlobalTensor<T> varRefGm_;

    TQue<QuePosition::VECIN, kBufferNum> updatesQueue_;
    TBuf<QuePosition::VECCALC> indicesWorkspaceBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;

    const ScatterAddWithSortedSimdTilingData* tilingData_ = nullptr;

    uint32_t blockIdx_ = 0;
    uint32_t rowCoreIdx_ = 0;
    uint32_t colCoreIdx_ = 0;

    int64_t colUbLoop_ = 0;
    int64_t normalLoopCols_ = 0;
    int64_t tailLoopCols_ = 0;
    int64_t colGmOffset_ = 0;

    constexpr static int32_t blockNumT_ = platform::GetUbBlockSize() / sizeof(T);
    constexpr static int32_t ubStride_ = platform::GetUbBlockSize() / sizeof(U); // 32B / sizeof(U)
};

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspace<T, U>::Init(
    GM_ADDR var, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn, const ScatterAddWithSortedSimdTilingData* tilingData)
{
    (void)var;

    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    rowCoreIdx_ = blockIdx_ / tilingData_->coreNumInCol;
    colCoreIdx_ = blockIdx_ % tilingData_->coreNumInCol;

    colGmOffset_ = static_cast<int64_t>(colCoreIdx_) * tilingData_->normalCoreColDetermNum;

    colUbLoop_ = (colCoreIdx_ == tilingData_->coreNumInColDeterm - 1) ? tilingData_->tailCoreColUbDetermLoop :
                                                                        tilingData_->normalCoreColUbDetermLoop;

    normalLoopCols_ = (colCoreIdx_ == tilingData_->coreNumInColDeterm - 1) ?
                          tilingData_->tailCoreNormalLoopDetermCols :
                          tilingData_->normalCoreNormalLoopDetermCols;

    tailLoopCols_ = (colCoreIdx_ == tilingData_->coreNumInColDeterm - 1) ? tilingData_->tailCoreTailLoopDetermCols :
                                                                           tilingData_->normalCoreTailLoopDetermCols;

    int64_t sumAreaBytes = static_cast<int64_t>(tilingData_->coreNumInRow) * kDouble * tilingData_->vecAlignSize;
    int64_t indexOffsetU = (sumAreaBytes + sizeof(U) - 1) / sizeof(U);

    indexWorkspace_.SetGlobalBuffer((__gm__ U*)workspace + indexOffsetU);
    sumWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);
    varRefGm_.SetGlobalBuffer((__gm__ T*)varRef);

    pipeIn.InitBuffer(updatesQueue_, kBufferNum, tilingData_->updatesDeterminBufferSize);
    pipeIn.InitBuffer(indicesWorkspaceBuf_, tilingData_->indicesWorkspaceBufferSize);
    pipeIn.InitBuffer(yBuf_, tilingData_->outBufferDeterminSize);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspace<T, U>::CopyInIndicesWorkspace(
    LocalTensor<U>& indicesWorkspaceLocal)
{
    DataCopyPadExtParams<U> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = 0;

    DataCopyExtParams ext;
    ext.blockCount = tilingData_->coreNumInRow;
    ext.blockLen = kDouble * sizeof(U);
    ext.srcStride = kCacheLineSize - kDouble * sizeof(U);
    ext.dstStride = 0;

    DataCopyPad(indicesWorkspaceLocal, indexWorkspace_, ext, pad);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspace<T, U>::CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyIn<T>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspace<T, U>::CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyOut<T>(dstGm, srcLocal, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspace<T, U>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    LocalTensor<U> indicesWorkspaceLocal = indicesWorkspaceBuf_.Get<U>();
    CopyInIndicesWorkspace(indicesWorkspaceLocal);

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(ev);
    WaitFlag<HardEvent::MTE2_S>(ev);

    for (int64_t col = 0; col < colUbLoop_; col++) {
        int64_t curLoopCols = (col == colUbLoop_ - 1) ? tailLoopCols_ : normalLoopCols_;
        int64_t curLoopColsAlign = (curLoopCols + blockNumT_ - 1) / blockNumT_ * blockNumT_;
        int64_t colOffset = colGmOffset_ + col * normalLoopCols_;

        U currentHead = indicesWorkspaceLocal.GetValue(ubStride_ * rowCoreIdx_ + 0);
        U currentTail = indicesWorkspaceLocal.GetValue(ubStride_ * rowCoreIdx_ + 1);

        U prevTail = static_cast<U>(-1);
        if (rowCoreIdx_ > 0) {
            prevTail = indicesWorkspaceLocal.GetValue(ubStride_ * (rowCoreIdx_ - 1) + 1);
        }

        if (currentHead != static_cast<U>(-1)) {
            bool handledByPrev = (rowCoreIdx_ > 0 && prevTail == currentHead);
            if (!handledByPrev) {
                uint64_t headOffset =
                    (static_cast<uint64_t>(rowCoreIdx_) * kDouble) * (tilingData_->vecAlignSize / sizeof(T)) +
                    static_cast<uint64_t>(colOffset);
                LocalTensor<T> upd = updatesQueue_.AllocTensor<T>();
                CopyIn(upd, sumWorkspace_, headOffset, 1, static_cast<uint32_t>(curLoopCols));
                updatesQueue_.EnQue(upd);
                upd = updatesQueue_.DeQue<T>();

                LocalTensor<T> yLocal = yBuf_.Get<T>();
                DataCopy(yLocal, upd, curLoopColsAlign);
                updatesQueue_.FreeTensor(upd);

                KernelUtil::WaitVToMte3();
                SetAtomicAdd<T>();
                uint64_t dstOffset = static_cast<uint64_t>(currentHead) * tilingData_->updatesInner + colOffset;
                CopyOut(varRefGm_, yLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
                SetAtomicNone();
                KernelUtil::WaitMte3ToV();
            }
        }

        if (currentTail != static_cast<U>(-1)) {
            bool handledByPrev = (rowCoreIdx_ > 0 && prevTail == currentTail);
            if (!handledByPrev) {
                uint64_t tailOffset =
                    (static_cast<uint64_t>(rowCoreIdx_) * kDouble + 1) * (tilingData_->vecAlignSize / sizeof(T)) +
                    static_cast<uint64_t>(colOffset);

                LocalTensor<T> upd = updatesQueue_.AllocTensor<T>();
                CopyIn(upd, sumWorkspace_, tailOffset, 1, static_cast<uint32_t>(curLoopCols));
                updatesQueue_.EnQue(upd);
                upd = updatesQueue_.DeQue<T>();

                LocalTensor<T> yLocal = yBuf_.Get<T>();
                DataCopy(yLocal, upd, curLoopColsAlign);
                updatesQueue_.FreeTensor(upd);

                int32_t nextCore = static_cast<int32_t>(rowCoreIdx_) + 1;
                U tailIdx = currentTail;

                while (nextCore < static_cast<int32_t>(tilingData_->coreNumInRow)) {
                    U nh = indicesWorkspaceLocal.GetValue(ubStride_ * nextCore + 0);
                    U nt = indicesWorkspaceLocal.GetValue(ubStride_ * nextCore + 1);

                    bool merged = false;
                    uint64_t off = 0;
                    if ((nh == static_cast<U>(-1) && nt == tailIdx) || (nh == tailIdx && nh != static_cast<U>(-1))) {
                        uint64_t coreMul = (nh == static_cast<U>(-1)) ?
                                               (static_cast<uint64_t>(nextCore) * kDouble + 1) :
                                               (static_cast<uint64_t>(nextCore) * kDouble);
                        off = coreMul * (tilingData_->vecAlignSize / sizeof(T)) + static_cast<uint64_t>(colOffset);
                        merged = true;
                    }
                    if (merged) {
                        LocalTensor<T> tmp = updatesQueue_.AllocTensor<T>();
                        CopyIn(tmp, sumWorkspace_, off, 1, static_cast<uint32_t>(curLoopCols));
                        updatesQueue_.EnQue(tmp);
                        tmp = updatesQueue_.DeQue<T>();

                        Add(yLocal, yLocal, tmp, curLoopCols);
                        updatesQueue_.FreeTensor(tmp);
                    }

                    if (!merged || nt == static_cast<U>(-1)) {
                        break;
                    }
                    nextCore++;
                }

                KernelUtil::WaitVToMte3();
                SetAtomicAdd<T>();
                uint64_t dstOffset = static_cast<uint64_t>(tailIdx) * tilingData_->updatesInner + colOffset;
                CopyOut(varRefGm_, yLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
                SetAtomicNone();
                KernelUtil::WaitMte3ToV();
            }
        }
    }
}

} // namespace ScatterAddWithSorted
#endif // SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_H