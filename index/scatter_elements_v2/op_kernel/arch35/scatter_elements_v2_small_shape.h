/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_SMALL_SHAPE_H_
#define SCATTER_ELEMENTS_V2_SMALL_SHAPE_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "scatter_elements_v2_small_shape_tiling.h"

namespace ScatterElements {
template <typename T, typename IndexT, uint32_t Rank, uint32_t Dim, bool SingleRow, uint32_t MaxThreads>
__simt_vf__ __aicore__ LAUNCH_BOUND(MaxThreads) inline void ScatterSmallShapeNone(
    __gm__ T* var, __gm__ const IndexT* indices, __gm__ const T* updates, uint32_t count, uint32_t dataStride,
    uint32_t indicesStride, uint32_t updatesStride, __gm__ const ScatterElementsV2SmallShapeTiling* td)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if constexpr (Rank == 1) {
            var[static_cast<uint32_t>(indices[i])] = updates[i];
        } else if constexpr (Rank == 2) {
            const uint32_t row = SingleRow ? 0 : i / indicesStride;
            const uint32_t col = i - row * indicesStride;
            uint32_t target;
            if constexpr (Dim == 0) {
                target = static_cast<uint32_t>(indices[i]) * dataStride + col;
            } else {
                target = row * dataStride + static_cast<uint32_t>(indices[i]);
            }
            var[target] = updates[row * updatesStride + col];
        } else {
            uint32_t remaining = i;
            uint32_t source = 0;
            uint32_t target = static_cast<uint32_t>(indices[i]) * td->axisStride;
#pragma unroll
            for (uint32_t d = 0; d < Rank - 1; ++d) {
                const uint32_t coordinate = remaining / td->indicesStrides[d];
                remaining -= coordinate * td->indicesStrides[d];
                source += coordinate * td->updatesStrides[d];
                target += coordinate * td->targetStrides[d];
            }
            source += remaining;
            target += remaining * td->targetStrides[Rank - 1];
            var[target] = updates[source];
        }
    }
}

template <typename T, typename IndexT, uint32_t Rank, uint32_t Dim = 0, bool SingleRow = false>
__aicore__ inline void LaunchScatterSmallShapeRank(__gm__ T* var, __gm__ const IndexT* indices, __gm__ const T* updates,
                                                   __gm__ const ScatterElementsV2SmallShapeTiling* td)
{
    constexpr uint32_t maxThreads = SMALL_SHAPE_THREAD_LIMITS[Rank - 1];
    uint32_t dataStride = 1;
    uint32_t indicesStride = 1;
    uint32_t updatesStride = 1;
    if constexpr (Rank == 2) {
        // Pass low-rank strides as scalar arguments so every SIMT thread does
        // not reload them from GM. The tiling layout and entry stay shared.
        dataStride = td->dataStride;
        indicesStride = td->indicesStride;
        updatesStride = td->updatesStride;
    }
    asc_vf_call<ScatterSmallShapeNone<T, IndexT, Rank, Dim, SingleRow, maxThreads>>(
        dim3(td->threads), var, indices, updates, td->allAxis, dataStride, indicesStride, updatesStride, td);
}

template <typename T, typename IndexT>
__aicore__ inline void LaunchScatterSmallShapeNone(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR tiling)
{
    auto td = reinterpret_cast<__gm__ const ScatterElementsV2SmallShapeTiling*>(tiling);
    auto dst = reinterpret_cast<__gm__ T*>(var);
    auto idx = reinterpret_cast<__gm__ const IndexT*>(indices);
    auto src = reinterpret_cast<__gm__ const T*>(updates);
    switch (td->rank) {
        case 1:
            LaunchScatterSmallShapeRank<T, IndexT, 1>(dst, idx, src, td);
            break;
        case 2:
            if (td->allAxis <= td->indicesStride && td->dim == 0) {
                LaunchScatterSmallShapeRank<T, IndexT, 2, 0, true>(dst, idx, src, td);
            } else if (td->allAxis <= td->indicesStride) {
                LaunchScatterSmallShapeRank<T, IndexT, 2, 1, true>(dst, idx, src, td);
            } else if (td->dim == 0) {
                LaunchScatterSmallShapeRank<T, IndexT, 2, 0>(dst, idx, src, td);
            } else {
                LaunchScatterSmallShapeRank<T, IndexT, 2, 1>(dst, idx, src, td);
            }
            break;
        case 3:
            LaunchScatterSmallShapeRank<T, IndexT, 3>(dst, idx, src, td);
            break;
        case 4:
            LaunchScatterSmallShapeRank<T, IndexT, 4>(dst, idx, src, td);
            break;
        case 5:
            LaunchScatterSmallShapeRank<T, IndexT, 5>(dst, idx, src, td);
            break;
        case 6:
            LaunchScatterSmallShapeRank<T, IndexT, 6>(dst, idx, src, td);
            break;
        case 7:
            LaunchScatterSmallShapeRank<T, IndexT, 7>(dst, idx, src, td);
            break;
        case 8:
            LaunchScatterSmallShapeRank<T, IndexT, 8>(dst, idx, src, td);
            break;
        default:
            break;
    }
}
} // namespace ScatterElements

#endif
