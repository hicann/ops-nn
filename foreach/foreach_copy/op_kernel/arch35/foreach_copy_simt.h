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
 * \file foreach_copy_simt.h
 * \brief SIMT kernel implementation for foreach_copy operator.
 *        ForeachCopyND (same-type copy) and ForeachCastND (cross-type cast).
 *        Uses per-core computation with prefix-sum-based flat-to-tensor index mapping.
 */

#ifndef FOREACH_COPY_SIMT_H
#define FOREACH_COPY_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "foreach_copy_tiling_data.h"
#include "foreach_copy_tiling_key.h"

namespace NsForeachCopy {

using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 2048;

// ========== Helper: get tensor address from ListTensorDesc ==========

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

// ========== ForeachCopyND: same-type copy kernel (MDE Section 5.2.1) ==========

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void ForeachCopyND(
    int32_t needCoreNum,
    int32_t tensorNum,
    int32_t tensorStart,
    int64_t tensorOffset,
    int64_t totalElements,
    GM_ADDR xList,
    GM_ADDR yList,
    __gm__ int64_t* numels)
{
    if (totalElements <= 0) {
        return;
    }

    // Per-core element range: [coreStart, coreEnd)
    int64_t perCoreElements = (totalElements + static_cast<int64_t>(needCoreNum) - 1) /
                              static_cast<int64_t>(needCoreNum);
    int64_t coreStart = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * perCoreElements;
    int64_t coreEnd = coreStart + perCoreElements;
    if (coreEnd > totalElements) {
        coreEnd = totalElements;
    }

    // Grid-stride loop within this core
    for (int64_t elemIdx = coreStart + static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
         elemIdx < coreEnd;
         elemIdx += static_cast<int64_t>(THREAD_NUM))
    {
        // Map flat element index to (tensorIdx, inTensorOffset) via prefix sum
        int32_t tensorIdx = tensorStart;
        int64_t prefix = 0;
        for (int32_t i = 0; i < tensorNum; ++i) {
            if (i < tensorStart) {
                prefix += numels[i];
                continue;
            }
            if (elemIdx < prefix + numels[i]) {
                tensorIdx = i;
                break;
            }
            prefix += numels[i];
        }
        int64_t inTensorOffset = elemIdx - prefix;

        // Perform copy
        if (inTensorOffset < numels[tensorIdx]) {
            __gm__ T* srcPtr = SimtGetTensorAddr<T>(xList, tensorIdx);
            __gm__ T* dstPtr = SimtGetTensorAddr<T>(yList, tensorIdx);
            dstPtr[inTensorOffset] = srcPtr[inTensorOffset];
        }
    }
}

// ========== ForeachCastND: cross-type cast kernel (MDE Section 5.2.2) ==========

template <typename SrcT, typename DstT>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void ForeachCastND(
    int32_t needCoreNum,
    int32_t tensorNum,
    int32_t tensorStart,
    int64_t tensorOffset,
    int64_t totalElements,
    GM_ADDR xList,
    GM_ADDR yList,
    __gm__ int64_t* numels)
{
    if (totalElements <= 0) {
        return;
    }

    // Per-core element range: [coreStart, coreEnd)
    int64_t perCoreElements = (totalElements + static_cast<int64_t>(needCoreNum) - 1) /
                              static_cast<int64_t>(needCoreNum);
    int64_t coreStart = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * perCoreElements;
    int64_t coreEnd = coreStart + perCoreElements;
    if (coreEnd > totalElements) {
        coreEnd = totalElements;
    }

    // Grid-stride loop within this core
    for (int64_t elemIdx = coreStart + static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
         elemIdx < coreEnd;
         elemIdx += static_cast<int64_t>(THREAD_NUM))
    {
        // Map flat element index to (tensorIdx, inTensorOffset) via prefix sum
        int32_t tensorIdx = tensorStart;
        int64_t prefix = 0;
        for (int32_t i = 0; i < tensorNum; ++i) {
            if (i < tensorStart) {
                prefix += numels[i];
                continue;
            }
            if (elemIdx < prefix + numels[i]) {
                tensorIdx = i;
                break;
            }
            prefix += numels[i];
        }
        int64_t inTensorOffset = elemIdx - prefix;

        // Perform cast copy
        if (inTensorOffset < numels[tensorIdx]) {
            __gm__ SrcT* srcPtr = SimtGetTensorAddr<SrcT>(xList, tensorIdx);
            __gm__ DstT* dstPtr = SimtGetTensorAddr<DstT>(yList, tensorIdx);
            dstPtr[inTensorOffset] = static_cast<DstT>(srcPtr[inTensorOffset]);
        }
    }
}

// ========== ProcessSame<T>: same-type copy entry (schMode 0-11) ==========

template <typename T>
__aicore__ inline void ProcessSame(GM_ADDR x, GM_ADDR y, GM_ADDR tilingBuf)
{
    __gm__ ForeachCopyTilingData* tilingData =
        reinterpret_cast<__gm__ ForeachCopyTilingData*>(tilingBuf);

    // totalElements already guarded in kernel entry

    int32_t needCoreNum = tilingData->needCoreNum;
    int32_t tensorNum = tilingData->tensorNum;
    int32_t tensorStart = tilingData->tensorStart;
    int64_t tensorOffset = tilingData->tensorOffset;
    int64_t totalElements = tilingData->totalElements;
    __gm__ int64_t* numels = tilingData->numels;

    AscendC::Simt::VF_CALL<ForeachCopyND<T>>(
        AscendC::Simt::Dim3(THREAD_NUM),
        needCoreNum, tensorNum, tensorStart, tensorOffset,
        totalElements, x, y, numels);
}

// ========== ProcessCast<SrcT, DstT>: cross-type cast entry (schMode 12-15) ==========

template <typename SrcT, typename DstT>
__aicore__ inline void ProcessCast(GM_ADDR x, GM_ADDR y, GM_ADDR tilingBuf)
{
    __gm__ ForeachCopyTilingData* tilingData =
        reinterpret_cast<__gm__ ForeachCopyTilingData*>(tilingBuf);

    int32_t needCoreNum = tilingData->needCoreNum;
    int32_t tensorNum = tilingData->tensorNum;
    int32_t tensorStart = tilingData->tensorStart;
    int64_t tensorOffset = tilingData->tensorOffset;
    int64_t totalElements = tilingData->totalElements;
    __gm__ int64_t* numels = tilingData->numels;

    AscendC::Simt::VF_CALL<ForeachCastND<SrcT, DstT>>(
        AscendC::Simt::Dim3(THREAD_NUM),
        needCoreNum, tensorNum, tensorStart, tensorOffset,
        totalElements, x, y, numels);
}

} // namespace NsForeachCopy

#endif // FOREACH_COPY_SIMT_H
