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
 * \file repeat_interleave_base.h
 * \brief
 */

#ifndef REPEAT_INTERLEAVE_BASE_H
#define REPEAT_INTERLEAVE_BASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "simt_api/asc_simt.h"

namespace RepeatInterleave {
using namespace AscendC;

constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_SEARCH = 128;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_PRECORESUM = 256;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_REPEAT = 2048;

template <typename U, typename V, typename AddrType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_SEARCH) inline void SimtSearchStartEnd(
    AddrType totalRepeats, AddrType repeatsNum, AddrType repeatsStart, AddrType repeatsEnd, __ubuf__ AddrType* tmpLocal,
    __gm__ V* prefixSumGm)
{
    AddrType repeatsPos = 0;
    if (threadIdx.x == 0) {
        repeatsPos = repeatsStart;
    } else {
        repeatsPos = repeatsEnd;
    }

    AddrType left = 0;
    AddrType right = totalRepeats - 1;

    while (left <= right) {
        AddrType mid = left + (right - left) / 2;
        if (prefixSumGm[mid] <= repeatsPos) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    if (threadIdx.x == 0) {
        tmpLocal[0] = left - 1;
        tmpLocal[1] = min(static_cast<AddrType>(prefixSumGm[left] - repeatsPos), repeatsNum);
    } else {
        tmpLocal[2] = left - 1;
        tmpLocal[3] = min(static_cast<AddrType>(repeatsPos + 1 - prefixSumGm[left - 1]), repeatsNum);
    }
}

template <typename T, typename U, typename V, typename AddrType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_REPEAT) inline void SimtRepeatSplitBatch(
    AddrType curBatchCount, AddrType totalRepeats, AddrType inputBatchOffset, AddrType outputBatchOffset,
    AddrType cpNum, __gm__ T* xGm, __gm__ U* repeatsGm, __gm__ volatile T* yGm, __gm__ V* prefixSumGm)
{
    for (AddrType batch = threadIdx.z; batch < curBatchCount; batch += blockDim.z) {
        AddrType curBatchXOffset = inputBatchOffset * batch;
        AddrType curBatchYOffset = outputBatchOffset * batch;
        for (AddrType repeatIdx = threadIdx.y; repeatIdx < totalRepeats; repeatIdx += blockDim.y) {
            AddrType curRepeatNum = repeatsGm[repeatIdx];
            T inputX = xGm[curBatchXOffset + repeatIdx * cpNum + threadIdx.x];
            V curOutRepeatOffset = prefixSumGm[repeatIdx];
            for (AddrType repeat = 0; repeat < curRepeatNum; repeat += 1) {
                yGm[curBatchYOffset + (curOutRepeatOffset + repeat) * cpNum + threadIdx.x] = inputX;
            }
        }
    }
}

template <typename T, typename U, typename V, typename AddrType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_REPEAT) inline void SimtRepeatSplitBatchCumSumUb(
    AddrType curBatchCount, AddrType totalRepeats, AddrType inputBatchOffset, AddrType outputBatchOffset,
    AddrType cpNum, __gm__ T* xGm, __gm__ U* repeatsGm, __gm__ volatile T* yGm, __ubuf__ V* cumSumLocal)
{
    for (AddrType batch = threadIdx.z; batch < curBatchCount; batch += blockDim.z) {
        AddrType curBatchXOffset = inputBatchOffset * batch;
        AddrType curBatchYOffset = outputBatchOffset * batch;
        for (AddrType repeatIdx = threadIdx.y; repeatIdx < totalRepeats; repeatIdx += blockDim.y) {
            AddrType curRepeatNum = repeatsGm[repeatIdx];
            T inputX = xGm[curBatchXOffset + repeatIdx * cpNum + threadIdx.x];
            V curOutRepeatOffset = cumSumLocal[repeatIdx];
            for (AddrType repeat = 0; repeat < curRepeatNum; repeat += 1) {
                yGm[curBatchYOffset + (curOutRepeatOffset + repeat) * cpNum + threadIdx.x] = inputX;
            }
        }
    }
}

template <typename T, typename U, typename V, typename AddrType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_REPEAT) inline void SimtSplitRepeats(
    AddrType startRepeatsIdx, AddrType endRepeatsIdx, U startRepeatsIdxResNum, U endRepeatsIdxResNum, AddrType cpNum,
    __gm__ T* xGm, __gm__ U* repeatsGm, __gm__ volatile T* yGm, __gm__ V* prefixSumGm)
{
    for (AddrType repeatIdx = threadIdx.y + startRepeatsIdx; repeatIdx <= endRepeatsIdx; repeatIdx += blockDim.y) {
        AddrType curRepeatNum;
        V curOutRepeatOffset;
        if (repeatIdx == startRepeatsIdx) {
            curRepeatNum = startRepeatsIdxResNum;
            curOutRepeatOffset = prefixSumGm[repeatIdx] + repeatsGm[repeatIdx] - curRepeatNum;
        } else if (repeatIdx == endRepeatsIdx) {
            curRepeatNum = endRepeatsIdxResNum;
            curOutRepeatOffset = prefixSumGm[repeatIdx];
        } else {
            curRepeatNum = repeatsGm[repeatIdx];
            curOutRepeatOffset = prefixSumGm[repeatIdx];
        }
        T inputX = xGm[repeatIdx * cpNum + threadIdx.x];
        for (AddrType repeat = 0; repeat < curRepeatNum; repeat += 1) {
            yGm[(curOutRepeatOffset + repeat) * cpNum + threadIdx.x] = inputX;
        }
    }
}

} // namespace RepeatInterleave
#endif // REPEAT_INTERLEAVE_BASE_H
