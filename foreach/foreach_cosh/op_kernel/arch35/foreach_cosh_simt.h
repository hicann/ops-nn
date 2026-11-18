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
 * \file foreach_cosh_simt.h
 * \brief SIMT kernel implementation for foreach_cosh
 *
 * The tensor list is treated as one flattened element range. Tiling assigns each core a
 * contiguous chunk [coreStart, coreEnd); a core then either
 *   - large segment: walks the intersecting tensors (binary search for the first one) and
 *     launches one VF per tensor slice, with an 8x per-thread unroll so that eight
 *     independent loads are in flight and the bandwidth-bound path covers the GM latency;
 *   - small segment (<= FLAT_SEG_MAX_ELEMENTS, e.g. lists of 0-dim scalars): launches a
 *     single VF for the whole segment in which every thread resolves its own tensor via
 *     the tiling prefix table, removing the serial per-tensor VF_CALL chain.
 */

#ifndef FOREACH_COSH_SIMT_H
#define FOREACH_COSH_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_cosh_tiling_data.h"
#include "foreach_cosh_tiling_key.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachCosh {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

// fixed thread count per core for every VF launch
constexpr uint32_t THREAD_NUM = 512;

// per-thread memory-level-parallelism unroll for the bandwidth-bound large segments:
// eight independent loads are issued back to back before the first store, growing the
// in-flight bytes per warp 8x so the stream can cover the GM access latency
constexpr int64_t UNROLL_COUNT = 8;

// segments no larger than this are handled by one flat VF whose threads resolve their own
// tensor through the prefix table: for such short segments the per-element lookup cost is
// negligible while the single launch removes the serial per-tensor VF_CALL chain that
// dominates small tensor lists (measured ~0.7us per tensor before this change)
constexpr int64_t FLAT_SEG_MAX_ELEMENTS = 4096;

/**
 * \brief Get the GM address of the idx-th tensor in a TensorList
 */
template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> TENSOR_PTR_SHIFT);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

/**
 * \brief Convert float16/bf16 to float32 for high precision calculation
 */
template <typename T>
__simt_callee__ inline float ToFloat32(T val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return static_cast<float>(val);
    } else {
        return val;
    }
}

/**
 * \brief Convert float32 back to float16/bf16
 */
template <typename T>
__simt_callee__ inline T FromFloat32(float val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return static_cast<bfloat16_t>(val);
    } else {
        return val;
    }
}

/**
 * \brief VF over one contiguous slice of a single tensor (large-segment path)
 *        cosh(x) = (exp(x) + exp(-x)) / 2, computed via coshf in float32
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachCoshTensorSegSimt(int32_t tensorId,
                                                                                     int64_t localStart, int64_t count,
                                                                                     GM_ADDR xList, GM_ADDR yList)
{
    // all threads issue the same two dependent loads, so the pointer table is fetched once
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    const int64_t stride = static_cast<int64_t>(AscendC::Simt::GetThreadNum());
    const int64_t end = localStart + count;
    const int64_t unrollEnd = end - (UNROLL_COUNT - 1) * stride;
    int64_t idx = localStart + static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
    for (; idx < unrollEnd; idx += UNROLL_COUNT * stride) {
        T val[UNROLL_COUNT];
#pragma unroll
        for (int64_t k = 0; k < UNROLL_COUNT; k++) {
            val[k] = xData[idx + k * stride];
        }
#pragma unroll
        for (int64_t k = 0; k < UNROLL_COUNT; k++) {
            yData[idx + k * stride] = FromFloat32<T>(coshf(ToFloat32<T>(val[k])));
        }
    }
    for (; idx < end; idx += stride) {
        yData[idx] = FromFloat32<T>(coshf(ToFloat32<T>(xData[idx])));
    }
}

/**
 * \brief VF over a whole small core segment, crossing tensor boundaries
 *        (small-list path): every thread locates its own tensor through the tiling
 *        prefix table, so a N-tensor list costs one VF launch instead of N
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachCoshFlatSegSimt(int64_t coreStart, int64_t coreEnd,
                                                                                   GM_ADDR xList, GM_ADDR yList,
                                                                                   GM_ADDR tiling)
{
    __gm__ const ForeachCoshTilingData* tilingGm = reinterpret_cast<__gm__ const ForeachCoshTilingData*>(tiling);
    const int32_t tensorCount = tilingGm->tensorCount;
    const int64_t stride = static_cast<int64_t>(AscendC::Simt::GetThreadNum());
    for (int64_t flat = coreStart + static_cast<int64_t>(AscendC::Simt::GetThreadIdx()); flat < coreEnd;
         flat += stride) {
        // locate the tensor owning this flat index: the last t with tensorStarts[t] <= flat.
        // Empty tensors hold equal starts, so the "last" rule skips them naturally.
        int32_t lo = 0;
        int32_t hi = tensorCount - 1;
        while (lo < hi) {
            const int32_t mid = (lo + hi + 1) >> 1;
            if (tilingGm->tensorStarts[mid] <= flat) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int64_t localIdx = flat - tilingGm->tensorStarts[lo];
        __gm__ T* xData = SimtGetTensorAddr<T>(xList, lo);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, lo);
        yData[localIdx] = FromFloat32<T>(coshf(ToFloat32<T>(xData[localIdx])));
    }
}

/**
 * \brief Process entry: per-core contiguous chunk of the flattened element range
 */
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const ForeachCoshTilingData* tilingData)
{
    const int64_t totalElements = tilingData->totalElements;
    if (totalElements <= 0) {
        return;
    }
    const int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int64_t coreStart = static_cast<int64_t>(blockIdx) * tilingData->perCoreElements;
    // The exclusive core split is only safe when the launched block count matches
    // needCoreNum; an extra core (block dim larger than needCoreNum, or a tiling whose
    // needCoreNum was squeezed below the platform core count) would otherwise binary-
    // search a flat index beyond totalElements and write outside the last tensor.
    if (blockIdx >= tilingData->needCoreNum || coreStart >= totalElements) {
        return;
    }
    int64_t coreEnd;
    if (blockIdx == tilingData->needCoreNum - 1) {
        coreEnd = coreStart + tilingData->lastCoreElements;
    } else {
        coreEnd = coreStart + tilingData->perCoreElements;
    }
    if (coreEnd > totalElements) {
        coreEnd = totalElements;
    }
    if (coreStart >= coreEnd) {
        return;
    }

    if (coreEnd - coreStart <= FLAT_SEG_MAX_ELEMENTS) {
        AscendC::Simt::VF_CALL<ForeachCoshFlatSegSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), coreStart, coreEnd, x, y,
                                                          tiling);
        return;
    }

    // large segment: binary search the first tensor intersecting [coreStart, coreEnd) on the
    // local tiling copy, then walk forward only over the intersecting ones
    int32_t lo = 0;
    int32_t hi = tilingData->tensorCount - 1;
    while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (tilingData->tensorStarts[mid] <= coreStart) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    for (int32_t t = lo; t < tilingData->tensorCount; t++) {
        const int64_t tensorStart = tilingData->tensorStarts[t];
        if (tensorStart >= coreEnd) {
            break;
        }
        const int64_t tensorEnd = tilingData->tensorStarts[t + 1];
        const int64_t segStart = (coreStart > tensorStart) ? coreStart : tensorStart;
        const int64_t segEnd = (coreEnd < tensorEnd) ? coreEnd : tensorEnd;
        if (segStart < segEnd) {
            AscendC::Simt::VF_CALL<ForeachCoshTensorSegSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), t,
                                                                segStart - tensorStart, segEnd - segStart, x, y);
        }
    }
}

} // namespace NsForeachCosh

#endif // FOREACH_COSH_SIMT_H
