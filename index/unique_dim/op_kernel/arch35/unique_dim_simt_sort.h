/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_DIM_SIMT_SORT_H
#define UNIQUE_DIM_SIMT_SORT_H

#include "unique_dim_simt.h"

// ============================================================
// Simt helper: Merge-path merge of GM ranges [start,mid) + [mid,end)
// Each thread binary-searches for its partition boundary (O(log n)),
// then does sequential merge within partition (O(n/K) per thread).
// Total: O(K*log(n) + n) comparisons vs O(n*log(n)) for per-element binary search.
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline void SimtMergePathImpl(
    int64_t globalTid, int64_t totalThreads,
    int64_t start, int64_t mid, int64_t end,
    __gm__ uint32_t *src, __gm__ uint32_t *dst,
    __gm__ T *inputFlat, int64_t rowLen)
{
    int64_t leftLen = mid - start;
    int64_t rightLen = end - mid;
    int64_t mergeLen = end - start;

    if (mergeLen <= 0) {
        return;
    }

    // Divide output among threads
    int64_t perThread = (mergeLen + totalThreads - 1) / totalThreads;
    int64_t myOutStart = globalTid * perThread;
    int64_t myOutEnd = (myOutStart + perThread < mergeLen) ? myOutStart + perThread : mergeLen;

    if (myOutStart >= mergeLen) {
        return;
    }

    // Step 1: Binary search for partition boundary at myOutStart
    int64_t lo = (myOutStart > rightLen) ? (myOutStart - rightLen) : 0;
    lo = (lo < leftLen) ? lo : leftLen;
    int64_t hi = (myOutStart < leftLen) ? myOutStart : leftLen;

    while (lo < hi) {
        int64_t m = lo + (hi - lo) / 2;
        int64_t j = myOutStart - m;
        int cmp = 0;
        if (j > 0) {
            cmp = SimtRowCompare<T>(inputFlat, rowLen,
                  static_cast<int64_t>(src[start + m]),
                  static_cast<int64_t>(src[mid + j - 1]));
        }
        if (j > 0 && cmp <= 0) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }

    int64_t iPos = lo;
    int64_t jPos = myOutStart - lo;

    // Step 2: Sequential merge from (iPos, jPos)
    for (int64_t outLocal = myOutStart; outLocal < myOutEnd; outLocal++) {
        if (iPos < leftLen && (jPos >= rightLen ||
            SimtRowCompare<T>(inputFlat, rowLen,
                static_cast<int64_t>(src[mid + jPos]),
                static_cast<int64_t>(src[start + iPos])) >= 0)) {
            dst[start + outLocal] = src[start + iPos];
            iPos++;
        } else {
            dst[start + outLocal] = src[mid + jPos];
            jPos++;
        }
    }
}

// ============================================================
// Simt helper: Merge one element within a UB tile merge round
// Computes binary search to find merge position, writes result to UB.
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline void SimtMergeOneUBElement(
    int32_t outIdx, int32_t pairSize, int32_t mergeWidth,
    int64_t tileLen, int32_t rOff, int32_t wOff,
    __gm__ T *inputFlat, int64_t rowLen,
    __local_mem__ int64_t *ubBuf)
{
    int32_t pairIdx = outIdx / pairSize;
    int32_t leftStart = pairIdx * pairSize;
    int32_t rightStart = leftStart + mergeWidth;
    int32_t localOut = outIdx - leftStart;

    int64_t leftLen = (mergeWidth < (tileLen - leftStart)) ? mergeWidth :
                      ((tileLen > leftStart) ? (tileLen - leftStart) : 0);
    int64_t rightLen = (mergeWidth < (tileLen - rightStart)) ? mergeWidth :
                       ((tileLen > rightStart) ? (tileLen - rightStart) : 0);

    if (localOut >= leftLen + rightLen) {
        ubBuf[wOff + outIdx] = -1;
        return;
    }

    int64_t lo = (localOut > rightLen) ? (localOut - rightLen) : 0;
    lo = (lo < leftLen) ? lo : leftLen;
    int64_t hi = (localOut < leftLen) ? localOut : leftLen;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t jIdx = localOut - mid;
        int cmp = 0;
        if (jIdx > 0 && (rightStart + jIdx - 1) < tileLen) {
            cmp = SimtRowCompare<T>(inputFlat, rowLen,
                ubBuf[rOff + leftStart + mid], ubBuf[rOff + rightStart + jIdx - 1]);
        }
        if (jIdx > 0 && cmp <= 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    int64_t iIdx = lo;
    int64_t jIdx2 = localOut - lo;

    if (iIdx < leftLen && (jIdx2 >= rightLen ||
        SimtRowCompare<T>(inputFlat, rowLen,
            ubBuf[rOff + rightStart + jIdx2], ubBuf[rOff + leftStart + iIdx]) >= 0)) {
        ubBuf[wOff + outIdx] = ubBuf[rOff + leftStart + iIdx];
    } else {
        ubBuf[wOff + outIdx] = ubBuf[rOff + rightStart + jIdx2];
    }
}

// ============================================================
// Simt helper: Cooperative merge rounds within a single tile in UB
// Runs log2(SIMT_TILE_SIZE/ITEMS_PER_THREAD) rounds of bitonic merge.
// Returns true if final result is in first half of UB, false if in second half.
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline bool SimtTileMergeRounds(
    int32_t tid, int32_t threadNum,
    int64_t tileLen,
    __gm__ T *inputFlat, int64_t rowLen,
    __local_mem__ int64_t *ubBuf)
{
    constexpr int32_t HALF = SIMT_TILE_SIZE;
    bool readFirst = true;

    for (int32_t mergeWidth = ITEMS_PER_THREAD; mergeWidth < SIMT_TILE_SIZE; mergeWidth *= 2) {
        int32_t pairSize = 2 * mergeWidth;
        int32_t rOff = readFirst ? 0 : HALF;
        int32_t wOff = readFirst ? HALF : 0;

        for (int32_t outIdx = tid; outIdx < SIMT_TILE_SIZE; outIdx += threadNum) {
            SimtMergeOneUBElement<T>(outIdx, pairSize, mergeWidth, tileLen, rOff, wOff,
                                      inputFlat, rowLen, ubBuf);
        }
        __syncthreads();
        readFirst = !readFirst;
    }

    return readFirst;
}

// ============================================================
// Simt helper: Sort a single tile cooperatively in UB
// Computes identity indices, runs sorting network, then
// cooperatively merges via double-buffered UB.
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline void SimtSortOneTile(
    int32_t tid, int32_t threadNum,
    int64_t tileLen, int64_t tileGlobalStart,
    __gm__ uint32_t *indices,
    __gm__ T *inputFlat, int64_t rowLen,
    __local_mem__ int64_t *ubBuf)
{
    constexpr int32_t HALF = SIMT_TILE_SIZE;

    // Each thread computes identity indices directly (no GM read)
    int64_t items[ITEMS_PER_THREAD];
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        int64_t idx = tid * ITEMS_PER_THREAD + j;
        items[j] = (idx < tileLen) ? (tileGlobalStart + idx) : -1;
    }

    // Sorting network for 2 elements (1 compare-and-swap)
    {
        int c = SimtRowCompare<T>(inputFlat, rowLen, items[0], items[1]);
        if (c > 0) { int64_t tmp = items[0]; items[0] = items[1]; items[1] = tmp; }
    }

    // Write sorted items to first half of UB
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        ubBuf[tid * ITEMS_PER_THREAD + j] = items[j];
    }
    __syncthreads();

    // Cooperative merge rounds
    bool readFirst = SimtTileMergeRounds<T>(tid, threadNum, tileLen, inputFlat, rowLen, ubBuf);

    // Write sorted result to GM
    int32_t finalOff = readFirst ? 0 : HALF;
    for (int32_t i = tid; i < tileLen; i += threadNum) {
        indices[tileGlobalStart + i] = static_cast<uint32_t>(ubBuf[finalOff + i]);
    }
}

// ============================================================
// Simt helper: Multi-round GM merge within a single core
// Merges sorted tiles from indices/sortBuf in successive doubling widths.
// ============================================================
template <typename T>
__simt_callee__ __aicore__ inline void SimtMergeTilesGM(
    int32_t tid, int32_t threadNum,
    int64_t coreStart, int64_t coreLen,
    int64_t tileSize,
    __gm__ uint32_t *indices, __gm__ uint32_t *sortBuf,
    __gm__ T *inputFlat, int64_t rowLen)
{
    int64_t width = tileSize;
    int32_t round = 0;

    while (width < coreLen) {
        for (int64_t s = 0; s < coreLen; s += 2 * width) {
            int64_t start = coreStart + s;
            int64_t mid = coreStart + ((s + width < coreLen) ? s + width : coreLen);
            int64_t end = coreStart + ((s + 2 * width < coreLen) ? s + 2 * width : coreLen);

            __gm__ uint32_t *src = (round % 2 == 0) ? indices : sortBuf;
            __gm__ uint32_t *dst = (round % 2 == 0) ? sortBuf : indices;

            if (mid >= end) {
                // Unmerged tail: copy from src to dst as-is
                for (int64_t i = tid; i < end - start; i += threadNum) {
                    dst[start + i] = src[start + i];
                }
                continue;
            }

            SimtMergePathImpl<T>(static_cast<int64_t>(tid), static_cast<int64_t>(threadNum),
                                  start, mid, end, src, dst, inputFlat, rowLen);
        }
        __syncthreads();
        width *= 2;
        round++;
    }

    // Copy back if odd rounds
    if (round % 2 == 1) {
        for (int64_t i = tid; i < coreLen; i += threadNum) {
            indices[coreStart + i] = sortBuf[coreStart + i];
        }
        __syncthreads();
    }
}

// ============================================================
// Simt Phase 1 All-in-One: Sort tiles in UB + multi-round GM merge
// in a single VF_CALL to avoid cross-VF_CALL GM cache visibility issues.
// Uses identity indices (no Phase0 dependency).
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM)
inline void SimtPhase1AllInOne(int64_t coreStart, int64_t coreLen,
                                int64_t tileSize, int64_t numLocalTiles,
                                __gm__ uint32_t *indices, __gm__ uint32_t *sortBuf,
                                __gm__ T *inputFlat, int64_t rowLen,
                                __local_mem__ int64_t *ubBuf)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();

    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;

    // Step 1: Sort each tile cooperatively in UB
    for (int64_t t = 0; t < numLocalTiles; t++) {
        int64_t tileOff = t * tileSize;
        int64_t tileLen = (tileOff + tileSize <= coreLen) ? tileSize : (coreLen - tileOff);
        int64_t tileGlobalStart = coreStart + tileOff;
        SimtSortOneTile<T>(tid, threadNum, tileLen, tileGlobalStart,
                           indices, inputFlat, rowLen, ubBuf);
    }

    // Ensure all tiles written to GM before merge reads them
    __syncthreads();

    // Step 2: Multi-round merge from GM
    if (numLocalTiles > 1) {
        SimtMergeTilesGM<T>(tid, threadNum, coreStart, coreLen, tileSize,
                            indices, sortBuf, inputFlat, rowLen);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

// ============================================================
// Simt all-core cooperative GM merge via merge path.
// All cores cooperate on a single pair [start, mid) + [mid, end).
// Each core's threads handle a disjoint subrange of the output.
// ============================================================
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_BLOCK_DIM)
inline void SimtMergePathPair(int64_t coreId, int64_t coreNum,
                               int64_t start, int64_t mid, int64_t end,
                               __gm__ uint32_t *src, __gm__ uint32_t *dst,
                               __gm__ T *inputFlat, int64_t rowLen)
{
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    int32_t tid = threadIdx.x;
    int32_t threadNum = blockDim.x;
    int64_t globalTid = coreId * static_cast<int64_t>(threadNum) + tid;
    int64_t totalThreads = coreNum * static_cast<int64_t>(threadNum);
    SimtMergePathImpl<T>(globalTid, totalThreads,
                          start, mid, end, src, dst, inputFlat, rowLen);
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

#endif // UNIQUE_DIM_SIMT_SORT_H
