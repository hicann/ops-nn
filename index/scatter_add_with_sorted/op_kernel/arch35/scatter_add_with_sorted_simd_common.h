/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_ADD_WITH_SORTED_SIMD_COMMON_H
#define SCATTER_ADD_WITH_SORTED_SIMD_COMMON_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

constexpr uint32_t kBufferNum = 2;       // double buffer
constexpr uint32_t kCacheLineSize = 128; // workspace index区按cacheline对齐
constexpr uint32_t kDouble = 2;          // workspace: head/tail 两行
constexpr uint32_t TMP_BUFFER_NUM = 2;
namespace KernelUtil {

__aicore__ inline void InitRowColTiling(
    const ScatterAddWithSortedSimdTilingData* tiling, uint32_t blockIdx, uint32_t& rowCoreIdx, uint32_t& colCoreIdx,
    bool& isStartRowCore, bool& isEndRowCore, int64_t& rowGmOffset, int64_t& colGmOffset, int64_t& rowUbLoop,
    int64_t& colUbLoop, int64_t& normalLoopRows, int64_t& tailLoopRows, int64_t& normalLoopCols, int64_t& tailLoopCols)
{
    rowCoreIdx = blockIdx / tiling->coreNumInCol;
    colCoreIdx = blockIdx % tiling->coreNumInCol;

    isStartRowCore = (rowCoreIdx == 0);
    isEndRowCore = (rowCoreIdx == tiling->coreNumInRow - 1);

    rowGmOffset = static_cast<int64_t>(rowCoreIdx) * tiling->normalCoreRowNum;
    colGmOffset = static_cast<int64_t>(colCoreIdx) * tiling->normalCoreColNum;

    rowUbLoop = (rowCoreIdx == tiling->coreNumInRow - 1) ? tiling->tailCoreRowUbLoop : tiling->normalCoreRowUbLoop;

    colUbLoop = (colCoreIdx == tiling->coreNumInCol - 1) ? tiling->tailCoreColUbLoop : tiling->normalCoreColUbLoop;

    normalLoopRows =
        (rowCoreIdx == tiling->coreNumInRow - 1) ? tiling->tailCoreNormalLoopRows : tiling->normalCoreNormalLoopRows;

    tailLoopRows =
        (rowCoreIdx == tiling->coreNumInRow - 1) ? tiling->tailCoreTailLoopRows : tiling->normalCoreTailLoopRows;

    normalLoopCols =
        (colCoreIdx == tiling->coreNumInCol - 1) ? tiling->tailCoreNormalLoopCols : tiling->normalCoreNormalLoopCols;

    tailLoopCols =
        (colCoreIdx == tiling->coreNumInCol - 1) ? tiling->tailCoreTailLoopCols : tiling->normalCoreTailLoopCols;
}

template <typename T>
__aicore__ inline void CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyPadExtParams<T> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = 0;

    DataCopyExtParams ext;
    ext.blockCount = nBurst;
    ext.blockLen = copyLen * sizeof(T);
    ext.srcStride = srcStride * sizeof(T);
    ext.dstStride = dstStride * sizeof(T);

    DataCopyPad(dstLocal, srcGm[offset], ext, pad);
}

template <typename T>
__aicore__ inline void CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams ext;
    ext.blockCount = nBurst;
    ext.blockLen = copyLen * sizeof(T);
    ext.srcStride = srcStride * sizeof(T);
    ext.dstStride = dstStride * sizeof(T);

    DataCopyPad(dstGm[offset], srcLocal, ext);
}

template <typename U>
__aicore__ inline void CopyInIndices(
    LocalTensor<U> dstLocal, GlobalTensor<U> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    CopyIn<U>(dstLocal, srcGm, offset, nBurst, copyLen, srcStride, dstStride);
}

template <typename T>
__aicore__ inline void BroadcastScalar(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint32_t count)
{
    T val = srcGm.GetValue(0);
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(ev);
    WaitFlag<HardEvent::S_V>(ev);
    Duplicate(dstLocal, val, count);
}

__aicore__ inline void WaitVToMte3()
{
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);
}

__aicore__ inline void WaitMte3ToV()
{
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(ev);
    WaitFlag<HardEvent::MTE3_V>(ev);
}

template <typename T>
__aicore__ inline void AtomicAddCopyOutSync(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t dstOffset, uint32_t copyLen)
{
    WaitVToMte3();
    SetAtomicAdd<T>();
    CopyOut<T>(dstGm, srcLocal, dstOffset, 1, copyLen);
    SetAtomicNone();
    WaitMte3ToV();
}

template <typename T, typename U>
__aicore__ inline bool InitSimdBase(
    const ScatterAddWithSortedSimdTilingData* tilingData, uint32_t blockIdx, uint32_t& rowCoreIdx, uint32_t& colCoreIdx,
    bool& isStartRowCore, bool& isEndRowCore, int64_t& rowGmOffset, int64_t& colGmOffset, int64_t& rowUbLoop,
    int64_t& colUbLoop, int64_t& normalLoopRows, int64_t& tailLoopRows, int64_t& normalLoopCols, int64_t& tailLoopCols,
    GlobalTensor<T>& updatesGm, GlobalTensor<U>& indicesGm, GM_ADDR updates, GM_ADDR indices)
{
    if (blockIdx >= tilingData->needCoreNum) {
        return false;
    }
    InitRowColTiling(
        tilingData, blockIdx, rowCoreIdx, colCoreIdx, isStartRowCore, isEndRowCore, rowGmOffset, colGmOffset, rowUbLoop,
        colUbLoop, normalLoopRows, tailLoopRows, normalLoopCols, tailLoopCols);
    updatesGm.SetGlobalBuffer((__gm__ T*)(updates));
    indicesGm.SetGlobalBuffer((__gm__ U*)(indices));
    return true;
}

template <typename T, typename U>
__aicore__ inline bool AccumulateOrInit(
    LocalTensor<T>& yLocal, LocalTensor<T>& updatesLocal, U& preId, U curId, int32_t curLoopCols,
    int32_t curLoopColsAlign)
{
    if (curId == preId) {
        Add(yLocal, yLocal, updatesLocal, curLoopCols);
        return false;
    }
    if (preId == static_cast<U>(-1)) {
        DataCopy(yLocal, updatesLocal, curLoopColsAlign);
        preId = curId;
        return false;
    }
    return true;
}

template <typename U, bool withPos>
__aicore__ inline void CopyInRowIndices(
    TQue<QuePosition::VECIN, kBufferNum>& indicesQueue, TQue<QuePosition::VECIN, kBufferNum>& posQueue,
    GlobalTensor<U>& indicesGm, GlobalTensor<U>& posGm, int64_t indicesGmOffset, uint32_t curLoopRows)
{
    LocalTensor<U> indicesLocal = indicesQueue.AllocTensor<U>();
    CopyInIndices<U>(indicesLocal, indicesGm, indicesGmOffset, 1, curLoopRows);
    indicesQueue.EnQue(indicesLocal);
    if constexpr (withPos) {
        LocalTensor<U> posLocal = posQueue.AllocTensor<U>();
        CopyInIndices<U>(posLocal, posGm, indicesGmOffset, 1, curLoopRows);
        posQueue.EnQue(posLocal);
    }
}

} // namespace KernelUtil
} // namespace ScatterAddWithSorted

#endif // SCATTER_ADD_WITH_SORTED_SIMD_COMMON_H