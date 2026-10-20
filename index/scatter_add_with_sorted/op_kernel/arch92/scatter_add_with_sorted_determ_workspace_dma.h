/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_DMA_H
#define SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_DMA_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../arch35/scatter_add_with_sorted_struct.h"
#include "../arch35/scatter_add_with_sorted_simd_common.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

template <typename T, typename U>
class ScatterAddWithSortedSimdDtermWorkspaceDMA {
public:
    __aicore__ inline ScatterAddWithSortedSimdDtermWorkspaceDMA(void) {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
                                const ScatterAddWithSortedSimdDMATilingData* tilingData);

    __aicore__ inline void CopyInIndicesWorkspace(LocalTensor<U>& indicesWorkspaceLocal);

    __aicore__ inline void CopyIn(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst,
                                  uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void CopyOut(GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst,
                                   uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0);

    __aicore__ inline void Process();

private:
    GlobalTensor<U> indexWorkspace_;
    GlobalTensor<T> sumWorkspace_;
    GlobalTensor<T> varRefGm_;

    TQue<QuePosition::VECIN, kBufferNum> updatesQueue_;
    TBuf<QuePosition::VECCALC> indicesWorkspaceBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;

    const ScatterAddWithSortedSimdDMATilingData* tilingData_ = nullptr;

    uint32_t blockIdx_ = 0;
    uint32_t rowCoreIdx_ = 0;
    uint32_t colCoreIdx_ = 0;

    int64_t tailLoopCols_ = 0;
    int64_t colGmOffset_ = 0;

    constexpr static int32_t blockNumT_ = platform::GetUbBlockSize() / sizeof(T);
    constexpr static int32_t ubStride_ = platform::GetUbBlockSize() / sizeof(U); // 32B / sizeof(U)
};

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspaceDMA<T, U>::Init(
    GM_ADDR var, GM_ADDR varRef, GM_ADDR workspace, TPipe& pipeIn,
    const ScatterAddWithSortedSimdDMATilingData* tilingData)
{
    (void)var;

    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    rowCoreIdx_ = blockIdx_ / tilingData_->colTileNum;
    colCoreIdx_ = blockIdx_ % tilingData_->colTileNum;

    colGmOffset_ = static_cast<int64_t>(colCoreIdx_) * tilingData_->normalCoreColDetermNum;

    tailLoopCols_ = (colCoreIdx_ == tilingData_->coreNumInColDeterm - 1) ? tilingData_->tailCoreTailLoopDetermCols :
                                                                           tilingData_->normalCoreTailLoopDetermCols;

    int64_t sumAreaBytes = static_cast<int64_t>(tilingData_->rowTileNum) * kDouble * tilingData_->vecAlignSize;
    int64_t indexOffsetU = (sumAreaBytes + sizeof(U) - 1) / sizeof(U);

    indexWorkspace_.SetGlobalBuffer((__gm__ U*)workspace + indexOffsetU);
    sumWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);
    varRefGm_.SetGlobalBuffer((__gm__ T*)varRef);

    pipeIn.InitBuffer(updatesQueue_, kBufferNum, tilingData_->updatesDeterminBufferSize);
    pipeIn.InitBuffer(indicesWorkspaceBuf_, tilingData_->indicesWorkspaceBufferSize);
    pipeIn.InitBuffer(yBuf_, tilingData_->outBufferDeterminSize);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspaceDMA<T, U>::CopyInIndicesWorkspace(
    LocalTensor<U>& indicesWorkspaceLocal)
{
    DataCopyPadExtParams<U> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = 0;

    DataCopyExtParams ext;
    ext.blockCount = tilingData_->rowTileNum;
    ext.blockLen = kDouble * sizeof(U);
    ext.srcStride = kCacheLineSize - kDouble * sizeof(U);
    ext.dstStride = 0;

    DataCopyPad(indicesWorkspaceLocal, indexWorkspace_, ext, pad);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspaceDMA<T, U>::CopyIn(LocalTensor<T> dstLocal,
                                                                               GlobalTensor<T> srcGm, uint64_t offset,
                                                                               uint32_t nBurst, uint32_t copyLen,
                                                                               uint32_t srcStride, uint32_t dstStride)
{
    KernelUtil::CopyIn<T>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspaceDMA<T, U>::CopyOut(GlobalTensor<T> dstGm,
                                                                                LocalTensor<T> srcLocal,
                                                                                uint64_t offset, uint32_t nBurst,
                                                                                uint32_t copyLen, uint32_t srcStride,
                                                                                uint32_t dstStride)
{
    KernelUtil::CopyOut<T>(dstGm, srcLocal, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T, typename U>
__aicore__ inline void ScatterAddWithSortedSimdDtermWorkspaceDMA<T, U>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    LocalTensor<U> indicesWorkspaceLocal = indicesWorkspaceBuf_.Get<U>();
    CopyInIndicesWorkspace(indicesWorkspaceLocal);

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(ev);
    WaitFlag<HardEvent::MTE2_S>(ev);

    int64_t curLoopCols = tailLoopCols_;
    int64_t curLoopColsAlign = (curLoopCols + blockNumT_ - 1) / blockNumT_ * blockNumT_;
    int64_t colOffset = colGmOffset_;

    U currentHead = indicesWorkspaceLocal.GetValue(ubStride_ * rowCoreIdx_ + 0);
    U currentTail = indicesWorkspaceLocal.GetValue(ubStride_ * rowCoreIdx_ + 1);

    U prevTail = static_cast<U>(-1);
    if (rowCoreIdx_ > 0) {
        prevTail = indicesWorkspaceLocal.GetValue(ubStride_ * (rowCoreIdx_ - 1) + 1);
    }

    if (currentTail != static_cast<U>(-1)) {
        bool handledByPrev = (rowCoreIdx_ > 0 && prevTail == currentTail);
        if (!handledByPrev) {
            uint64_t tailOffset = (static_cast<uint64_t>(rowCoreIdx_) * kDouble + 1) *
                                      (tilingData_->vecAlignSize / sizeof(T)) +
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

            while (nextCore < static_cast<int32_t>(tilingData_->rowTileNum)) {
                U nh = indicesWorkspaceLocal.GetValue(ubStride_ * nextCore + 0);
                U nt = indicesWorkspaceLocal.GetValue(ubStride_ * nextCore + 1);

                bool merged = false;
                bool merged_ext = false;
                uint64_t off = 0;
                if ((nh == static_cast<U>(-1) && nt == tailIdx) ||
                    (nh == tailIdx && nh != nt)) { // 首尾相同放尾 或者 下个头等于当前尾
                    uint64_t coreMul = (nh == static_cast<U>(-1)) ? (static_cast<uint64_t>(nextCore) * kDouble + 1) :
                                                                    (static_cast<uint64_t>(nextCore) * kDouble);
                    off = coreMul * (tilingData_->vecAlignSize / sizeof(T)) + static_cast<uint64_t>(colOffset);
                    merged = true;
                } else if (nh == tailIdx && nh == nt) {
                    uint64_t offPos0 = (static_cast<uint64_t>(nextCore) * kDouble) *
                                           (tilingData_->vecAlignSize / sizeof(T)) +
                                       static_cast<uint64_t>(colOffset);
                    uint64_t offPos1 = (static_cast<uint64_t>(nextCore) * kDouble + 1) *
                                           (tilingData_->vecAlignSize / sizeof(T)) +
                                       static_cast<uint64_t>(colOffset);
                    LocalTensor<T> tmp = updatesQueue_.AllocTensor<T>();
                    CopyIn(tmp, sumWorkspace_, offPos0, 1, static_cast<uint32_t>(curLoopCols));
                    updatesQueue_.EnQue(tmp);
                    tmp = updatesQueue_.DeQue<T>();

                    Add(yLocal, yLocal, tmp, curLoopCols);
                    updatesQueue_.FreeTensor(tmp);

                    LocalTensor<T> tmp1 = updatesQueue_.AllocTensor<T>();
                    CopyIn(tmp1, sumWorkspace_, offPos1, 1, static_cast<uint32_t>(curLoopCols));
                    updatesQueue_.EnQue(tmp1);
                    tmp1 = updatesQueue_.DeQue<T>();

                    Add(yLocal, yLocal, tmp1, curLoopCols);
                    updatesQueue_.FreeTensor(tmp1);
                    merged_ext = true;
                }
                if (merged) {
                    LocalTensor<T> tmp = updatesQueue_.AllocTensor<T>();
                    CopyIn(tmp, sumWorkspace_, off, 1, static_cast<uint32_t>(curLoopCols));
                    updatesQueue_.EnQue(tmp);
                    tmp = updatesQueue_.DeQue<T>();

                    Add(yLocal, yLocal, tmp, curLoopCols);
                    updatesQueue_.FreeTensor(tmp);
                }

                if (!(merged || merged_ext) || nt == static_cast<U>(-1)) {
                    break;
                }
                nextCore++;
            }

            KernelUtil::WaitVToMte3();
            SetAtomicAdd<T>();
            uint64_t dstOffset = static_cast<uint64_t>(tailIdx) * tilingData_->varShape[1] + colOffset;
            CopyOut(varRefGm_, yLocal, dstOffset, 1, static_cast<uint32_t>(curLoopCols));
            SetAtomicNone();
            KernelUtil::WaitMte3ToV();
        }
    }
}

} // namespace ScatterAddWithSorted
#endif // SCATTER_ADD_WITH_SORTED_SIMD_DETERM_WORKSPACE_DMA_H
