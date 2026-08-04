/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_sub.cpp
 * \brief inplace_sub
 */

#include "kernel_operator.h"
#include "inplace_sub_tiling_data.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"

namespace {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_callee__ __aicore__ inline T SubValue(T lhs, T rhs)
{
    return static_cast<T>(lhs - rhs);
}

__simt_callee__ __aicore__ inline int32_t NormalizeIndex(int32_t index, int32_t n)
{
    int64_t normalized = static_cast<int64_t>(index) % static_cast<int64_t>(n);
    if (normalized < 0) {
        normalized += static_cast<int64_t>(n);
    }
    return static_cast<int32_t>(normalized);
}

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void InplaceSubCompute(
    int32_t needCoreNum, int32_t coreId, int32_t n, int32_t k, int64_t rowSize, int32_t perCoreN, __gm__ T* xGm,
    __gm__ int32_t* indicesGm, __gm__ T* vGm, __gm__ T* yGm)
{
    if (coreId >= needCoreNum) {
        return;
    }
    int32_t startRow = coreId * perCoreN;
    int32_t endRow = startRow + perCoreN;
    if (endRow > n) {
        endRow = n;
    }
    if (startRow >= endRow) {
        return;
    }

    int64_t startElem = static_cast<int64_t>(startRow) * rowSize;
    int64_t endElem = static_cast<int64_t>(endRow) * rowSize;
    for (int64_t idx = startElem + static_cast<int64_t>(threadIdx.x); idx < endElem;
         idx += static_cast<int64_t>(THREAD_NUM)) {
        yGm[idx] = xGm[idx];
    }

    for (int32_t kIdx = 0; kIdx < k; ++kIdx) {
        int32_t dstRow = NormalizeIndex(indicesGm[kIdx], n);
        if (dstRow < startRow || dstRow >= endRow) {
            continue;
        }
        int64_t vBase = static_cast<int64_t>(kIdx) * rowSize;
        int64_t yBase = static_cast<int64_t>(dstRow) * rowSize;
        for (int64_t offset = static_cast<int64_t>(threadIdx.x); offset < rowSize;
             offset += static_cast<int64_t>(THREAD_NUM)) {
            yGm[yBase + offset] = SubValue(yGm[yBase + offset], vGm[vBase + offset]);
        }
    }
}

template <typename T, int64_t ELEMENTS_PER_VALUE = 1>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR indices, GM_ADDR v, GM_ADDR y, const InplaceSubTilingData* tilingData)
{
    __gm__ T* xGm = reinterpret_cast<__gm__ T*>(x);
    __gm__ int32_t* indicesGm = reinterpret_cast<__gm__ int32_t*>(indices);
    __gm__ T* vGm = reinterpret_cast<__gm__ T*>(v);
    __gm__ T* yGm = reinterpret_cast<__gm__ T*>(y);
    int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    int64_t rowSize = tilingData->innerSize * ELEMENTS_PER_VALUE;

    asc_vf_call<InplaceSubCompute<T>>(dim3(THREAD_NUM), tilingData->needCoreNum, blockIdx, tilingData->n, tilingData->k,
                                      rowSize, tilingData->perCoreN, xGm, indicesGm, vGm, yGm);
}

} // namespace

extern "C" __global__ __aicore__ void inplace_sub(GM_ADDR x, GM_ADDR indices, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(InplaceSubTilingData);
    GET_TILING_DATA_WITH_STRUCT(InplaceSubTilingData, tilingData, tiling);
    if (tilingData.needCoreNum == 0 || tilingData.n <= 0 || tilingData.innerSize <= 0) {
        return;
    }
#if ORIG_DTYPE_X == DT_COMPLEX32
    Process<half, 2>(x, indices, v, y, &tilingData);
#elif ORIG_DTYPE_X == DT_COMPLEX64
    Process<float, 2>(x, indices, v, y, &tilingData);
#else
    Process<DTYPE_X>(x, indices, v, y, &tilingData);
#endif
}
