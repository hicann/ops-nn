/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_DIM_SIMT_H
#define UNIQUE_DIM_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t SIMT_BLOCK_DIM = 256;
constexpr int32_t ITEMS_PER_THREAD = 2;
constexpr int32_t SIMT_TILE_SIZE = SIMT_BLOCK_DIM * ITEMS_PER_THREAD; // 512
constexpr int32_t GM_BLOCK_DIM = 2048;                                // more threads for GM-bound ops (transpose, copy)

// ============================================================
// Scalar compare helper: bf16/double use totalOrder via unsigned integer
// ============================================================

// TotalOrder comparison for floating-point types that need it (bf16, double)
// Normalizes -0 → +0 so they compare equal (stable sort preserves first-appearing one)
template <typename T, typename U>
__simt_callee__ __aicore__ inline int SimtCompareTotalOrder(T lhs, T rhs, U signBit)
{
    U lu = *reinterpret_cast<const U*>(&lhs);
    U ru = *reinterpret_cast<const U*>(&rhs);
    if (lu == signBit)
        lu = 0; // -0 → +0
    if (ru == signBit)
        ru = 0;
    if (lu & signBit) {
        lu = ~lu;
    } else {
        lu ^= signBit;
    }
    if (ru & signBit) {
        ru = ~ru;
    } else {
        ru ^= signBit;
    }
    if (lu < ru)
        return -1;
    if (lu > ru)
        return 1;
    return 0;
}

// NaN detection via bit pattern: exponent all-ones && mantissa non-zero
template <typename T>
__simt_callee__ __aicore__ inline bool SimtIsNan(T val)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        uint16_t u = *reinterpret_cast<const uint16_t*>(&val);
        return (u & 0x7F80U) == 0x7F80U && (u & 0x007FU) != 0;
    } else if constexpr (IsSameType<T, half>::value) {
        uint16_t u = *reinterpret_cast<const uint16_t*>(&val);
        return (u & 0x7C00U) == 0x7C00U && (u & 0x03FFU) != 0;
    } else if constexpr (IsSameType<T, float>::value) {
        uint32_t u = *reinterpret_cast<const uint32_t*>(&val);
        return (u & 0x7F800000U) == 0x7F800000U && (u & 0x007FFFFFU) != 0;
    } else if constexpr (IsSameType<T, double>::value) {
        uint64_t u = *reinterpret_cast<const uint64_t*>(&val);
        return (u & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL && (u & 0x000FFFFFFFFFFFFFULL) != 0;
    } else {
        return false;
    }
}

// Version for simt context (called from __simt_callee__/__simt_vf__)
// NOTE: NaN compares as equal here (for stable sorting). NaN dedup is handled
// separately in SimtAdjacentDiff to keep sort order consistent with CPU.
template <typename T>
__simt_callee__ __aicore__ inline int SimtScalarCompare(T lhs, T rhs)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        return SimtCompareTotalOrder<T, uint16_t>(lhs, rhs, 0x8000U);
    } else if constexpr (IsSameType<T, double>::value) {
        return SimtCompareTotalOrder<T, uint64_t>(lhs, rhs, 0x8000000000000000ULL);
    } else {
        if (lhs < rhs)
            return -1;
        if (lhs > rhs)
            return 1;
        return 0;
    }
}

// ============================================================
// Simt helper: Row comparison with sentinel support
// Returns -1 if a<b, 0 if equal, 1 if a>b
// Sentinel -1 sorts last (greater than any real index)
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline int SimtRowCompare(__gm__ T* inputFlat, int64_t rowLen, int64_t a, int64_t b)
{
    if (a == -1 && b == -1) {
        return 0;
    }
    if (a == -1) {
        return 1;
    }
    if (b == -1) {
        return -1;
    }
    for (int64_t j = 0; j < rowLen; ++j) {
        T lhs = inputFlat[a * rowLen + j];
        T rhs = inputFlat[b * rowLen + j];
        int cmp = SimtScalarCompare<T>(lhs, rhs);
        if (cmp != 0) {
            return cmp;
        }
    }
    return 0;
}

// Dedup variant: same single-pass comparison, but also treats NaN as not-equal
// (for sorting NaN compares equal to preserve stable order; for dedup NaN must be unique)
template <typename T>
__simt_callee__ __aicore__ inline int SimtRowCompareDedup(__gm__ T* dataPtr, int64_t rowSize, int64_t rowA,
                                                          int64_t rowB)
{
    if (rowA == -1 && rowB == -1) {
        return 0;
    }
    if (rowA == -1) {
        return 1;
    }
    if (rowB == -1) {
        return -1;
    }
    for (int64_t colIdx = 0; colIdx < rowSize; ++colIdx) {
        T valA = dataPtr[rowA * rowSize + colIdx];
        T valB = dataPtr[rowB * rowSize + colIdx];
        int cmpResult = SimtScalarCompare<T>(valA, valB);
        if (cmpResult != 0) {
            return cmpResult;
        }
        if (SimtIsNan<T>(valA) || SimtIsNan<T>(valB)) {
            return 1;
        }
    }
    return 0;
}

// ============================================================
// Simt Phase 0: Initialize indices [0, 1, ..., numInp-1]
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtInitIndices(int64_t coreStart, int64_t coreLen,
                                                                                __gm__ uint32_t* indices)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        indices[i] = static_cast<uint32_t>(i);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt copy utility: dst[i] = src[i] for i in [start, end)
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtCopyRange(int64_t start, int64_t end,
                                                                              __gm__ uint32_t* src,
                                                                              __gm__ uint32_t* dst)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = start + tid; i < end; i += threadNum) {
        dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt transpose: [outerSize, dkSize, innerSize] → [dkSize, outerSize, innerSize]
// Each core processes [coreStart, coreStart+coreLen) in the flat index space.
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(GM_BLOCK_DIM) inline void SimtTransposeDim(int64_t coreStart, int64_t coreLen,
                                                                               int64_t outerSize, int64_t dkSize,
                                                                               int64_t innerSize, __gm__ T* src,
                                                                               __gm__ T* dst)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    int64_t dkInner = dkSize * innerSize;

    for (int64_t flat = coreStart + tid; flat < coreStart + coreLen; flat += threadNum) {
        int64_t j = flat / dkInner;
        int64_t rem = flat % dkInner;
        int64_t i = rem / innerSize;
        int64_t l = rem % innerSize;
        dst[i * outerSize * innerSize + j * innerSize + l] = src[flat];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt flat copy: dst[i] = src[i] for i in [coreStart, coreStart+coreLen)
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(GM_BLOCK_DIM) inline void SimtCopyFlat(int64_t coreStart, int64_t coreLen,
                                                                           __gm__ T* src, __gm__ T* dst)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Phase 3: Adjacent diff
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtAdjacentDiff(int64_t coreStart, int64_t coreLen,
                                                                                 __gm__ uint32_t* indices,
                                                                                 __gm__ uint32_t* flags,
                                                                                 __gm__ T* inputFlat, int64_t rowLen)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        if (i == 0) {
            flags[0] = 1;
        } else {
            int64_t prevKey = static_cast<int64_t>(indices[i - 1]);
            int64_t curKey = static_cast<int64_t>(indices[i]);
            int cmp = SimtRowCompareDedup<T>(inputFlat, rowLen, prevKey, curKey);
            flags[i] = (cmp != 0) ? 1 : 0;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Phase 4 Step 1: Local inclusive scan within core
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtLocalScan(int64_t coreStart, int64_t coreLen,
                                                                              __gm__ uint32_t* flags, int64_t coreId,
                                                                              __gm__ uint32_t* partialSum,
                                                                              int64_t psStride)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    if (tid == 0) {
        int64_t runningSum = 0;
        for (int64_t i = 0; i < coreLen; i++) {
            runningSum += static_cast<int64_t>(flags[coreStart + i]);
            flags[coreStart + i] = static_cast<uint32_t>(runningSum);
        }
        partialSum[coreId * psStride] = static_cast<uint32_t>(runningSum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Phase 4 Step 4: Add global prefix to local scan results
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtAddGlobalPrefix(int64_t coreStart, int64_t coreLen,
                                                                                    __gm__ uint32_t* flags,
                                                                                    int64_t coreId,
                                                                                    __gm__ uint32_t* globalPrefix)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    int64_t myPrefix = static_cast<int64_t>(globalPrefix[coreId]);
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        flags[i] = flags[i] + static_cast<uint32_t>(myPrefix);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Phase 5: Scatter inverse indices
// Reads uint32_t workspace, writes int64_t output
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtScatterInverse(int64_t coreStart, int64_t coreLen,
                                                                                   __gm__ uint32_t* sortedBackup,
                                                                                   __gm__ uint32_t* flags,
                                                                                   __gm__ int64_t* inverseOut)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        int64_t origIdx = static_cast<int64_t>(sortedBackup[i]);
        int64_t groupIdx = static_cast<int64_t>(flags[i]) - 1;
        inverseOut[origIdx] = groupIdx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Phase 6: Unique detect
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtUniqueDetect(
    int64_t coreStart, int64_t coreLen, __gm__ uint32_t* indices, __gm__ uint32_t* flags, __gm__ uint32_t* compactKeys,
    __gm__ uint32_t* positions, int64_t returnCounts)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = coreStart + tid; i < coreStart + coreLen; i += threadNum) {
        int64_t curGroup = static_cast<int64_t>(flags[i]);
        int64_t prevGroup = (i == 0) ? 0 : static_cast<int64_t>(flags[i - 1]);
        if (curGroup != prevGroup) {
            int64_t origIdx = static_cast<int64_t>(indices[i]);
            int64_t compactPos = curGroup - 1;
            compactKeys[compactPos] = static_cast<uint32_t>(origIdx);
            if (returnCounts != 0) {
                positions[compactPos] = static_cast<uint32_t>(i);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Gather rows
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(GM_BLOCK_DIM) inline void SimtGatherRows(int64_t eStart, int64_t eLen,
                                                                             __gm__ uint32_t* compactKeys,
                                                                             __gm__ T* inputFlat, __gm__ T* valueOut,
                                                                             int64_t rowLen)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    // The flat range [eStart, eStart + eLen) is contiguous in output space and
    // spans a whole number of rows plus at most two partial rows at the ends.
    // Iterate row-by-row so compactKeys[outIdx] is read once per row and the
    // inner loop walks j sequentially (no per-element div/mod).
    int64_t eEnd = eStart + eLen;
    int64_t firstOut = eStart / rowLen;
    int64_t lastOut = (eEnd - 1) / rowLen;
    for (int64_t outIdx = firstOut; outIdx <= lastOut; outIdx++) {
        int64_t rowBase = outIdx * rowLen;
        int64_t jLo = (rowBase < eStart) ? (eStart - rowBase) : 0;
        int64_t jHi = (rowBase + rowLen > eEnd) ? (eEnd - rowBase) : rowLen;
        int64_t srcIdx = static_cast<int64_t>(compactKeys[outIdx]);
        for (int64_t j = jLo + tid; j < jHi; j += threadNum) {
            valueOut[rowBase + j] = inputFlat[srcIdx * rowLen + j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt Compute Counts
// Reads uint32_t positions, writes int64_t counts output
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtComputeCounts(int64_t cntStart, int64_t cntEnd,
                                                                                  __gm__ uint32_t* positions,
                                                                                  __gm__ int64_t* countsOut)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    for (int64_t i = cntStart + tid; i < cntEnd; i += threadNum) {
        countsOut[i] = static_cast<int64_t>(positions[i + 1]) - static_cast<int64_t>(positions[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt global prefix scan (Phase 4 Step 2)
// Thread 0 scans partialSum, writes globalPrefix and totalUnique
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtGlobalPrefixScan(int64_t coreNum, int64_t psStride,
                                                                                     __gm__ uint32_t* partialSum,
                                                                                     __gm__ uint32_t* globalPrefix)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    if (tid == 0) {
        int64_t prefix = 0;
        for (int64_t c = 0; c < coreNum; c++) {
            int64_t val = static_cast<int64_t>(partialSum[c * psStride]);
            globalPrefix[c] = static_cast<uint32_t>(prefix);
            prefix += val;
        }
        globalPrefix[coreNum] = static_cast<uint32_t>(prefix);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt write single uint32_t value
// ============================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM) inline void SimtWriteValue(__gm__ uint32_t* dst, int64_t offset,
                                                                               uint32_t value)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    if (tid == 0) {
        dst[offset] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

#endif // UNIQUE_DIM_SIMT_H
