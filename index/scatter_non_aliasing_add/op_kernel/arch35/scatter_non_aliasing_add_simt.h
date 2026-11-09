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
 * \file scatter_non_aliasing_add_simt.h
 * \brief SIMT kernel for scatter_non_aliasing_add
 */

#ifndef SCATTER_NON_ALIASING_ADD_SIMT_H
#define SCATTER_NON_ALIASING_ADD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "scatter_non_aliasing_add_tiling_data.h"
#include "scatter_non_aliasing_add_tiling_key.h"

namespace NsScatterNonAliasingAdd {

using namespace AscendC;

template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? 1024 : 512;

template <typename T, typename IndexT, typename IDX_T>
__simt_callee__ inline IDX_T ComputeScatterOffset(int32_t K, IDX_T i, __gm__ IndexT* indices,
                                                  __ubuf__ int64_t* stridesUb)
{
    IDX_T offset = 0;
    for (int32_t k = 0; k < K; k++) {
        IDX_T indexVal = static_cast<IDX_T>(indices[i * static_cast<IDX_T>(K) + k]);
        offset += indexVal * static_cast<IDX_T>(stridesUb[k]);
    }
    return offset;
}

template <typename T, typename IndexT, typename IDX_T>
__simt_callee__ inline void ScatterAddSlice(IDX_T offset, IDX_T i, IDX_T updateDataNum, __gm__ T* updates, __gm__ T* y)
{
    for (IDX_T j = static_cast<IDX_T>(threadIdx.x); j < updateDataNum; j += static_cast<IDX_T>(blockDim.x)) {
        T yVal = y[offset + j];
        T updateVal = updates[i * updateDataNum + j];
        y[offset + j] = yVal + updateVal;
    }
}

template <typename T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpScatterCopySimtKernel(int64_t totalElements,
                                                                                             int64_t perCoreElements,
                                                                                             __gm__ T* x, __gm__ T* y)
{
    int64_t coreStart64 = static_cast<int64_t>(blockIdx.x) * perCoreElements;
    int64_t coreEnd64 = coreStart64 + perCoreElements;
    if (coreEnd64 > totalElements) {
        coreEnd64 = totalElements;
    }
    IDX_T coreStart = static_cast<IDX_T>(coreStart64);
    IDX_T coreEnd = static_cast<IDX_T>(coreEnd64);
    for (IDX_T idx = coreStart + static_cast<IDX_T>(threadIdx.x); idx < coreEnd;
         idx += static_cast<IDX_T>(blockDim.x)) {
        y[idx] = x[idx];
    }
}

template <typename T, typename IndexT, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpScatterAddSimtKernel(
    int32_t K, int64_t totalElements, int64_t totalScatters, int64_t updateDataNum, int64_t perCoreElements,
    __ubuf__ int64_t* stridesUb, __gm__ IndexT* indices, __gm__ T* updates, __gm__ T* y)
{
    IDX_T totalScat = static_cast<IDX_T>(totalScatters);
    IDX_T updateNum = static_cast<IDX_T>(updateDataNum);

    int64_t coreStart64 = static_cast<int64_t>(blockIdx.x) * perCoreElements;
    int64_t coreEnd64 = coreStart64 + perCoreElements;
    if (coreEnd64 > totalElements) {
        coreEnd64 = totalElements;
    }
    IDX_T coreStart = static_cast<IDX_T>(coreStart64);
    IDX_T coreEnd = static_cast<IDX_T>(coreEnd64);

    for (IDX_T i = 0; i < totalScat; i++) {
        IDX_T offset = ComputeScatterOffset<T, IndexT, IDX_T>(K, i, indices, stridesUb);
        if (offset >= coreStart && offset < coreEnd) {
            ScatterAddSlice<T, IndexT, IDX_T>(offset, i, updateNum, updates, y);
        }
    }
}

__aicore__ static inline bool UseInt32Index(const ScatterNonAliasingAddTilingData& tilingData)
{
    if (tilingData.totalElements > static_cast<int64_t>(INT32_MAX) ||
        tilingData.perCoreElements > static_cast<int64_t>(INT32_MAX) ||
        tilingData.totalScatters > static_cast<int64_t>(INT32_MAX) ||
        tilingData.updateDataNum > static_cast<int64_t>(INT32_MAX)) {
        return false;
    }
    if (tilingData.totalScatters > 0) {
        if (tilingData.K > 0 &&
            tilingData.totalScatters > static_cast<int64_t>(INT32_MAX) / static_cast<int64_t>(tilingData.K)) {
            return false;
        }
        if (tilingData.updateDataNum > 0 &&
            tilingData.totalScatters > static_cast<int64_t>(INT32_MAX) / tilingData.updateDataNum) {
            return false;
        }
    }
    return true;
}

template <typename T, typename IndexT>
__aicore__ inline void ScatterDispatch(const ScatterNonAliasingAddTilingData& tilingData, __ubuf__ int64_t* stridesUb,
                                       __gm__ T* xGm, __gm__ IndexT* indicesGm, __gm__ T* updatesGm, __gm__ T* yGm)
{
    if (UseInt32Index(tilingData)) {
        asc_vf_call<OpScatterCopySimtKernel<T, int32_t>>(dim3(THREADS<int32_t>), tilingData.totalElements,
                                                         tilingData.perCoreElements, xGm, yGm);
        GlobalTensor<T> yFlushGm;
        yFlushGm.SetGlobalBuffer(yGm);
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yFlushGm);
        asc_vf_call<OpScatterAddSimtKernel<T, IndexT, int32_t>>(
            dim3(THREADS<int32_t>), tilingData.K, tilingData.totalElements, tilingData.totalScatters,
            tilingData.updateDataNum, tilingData.perCoreElements, stridesUb, indicesGm, updatesGm, yGm);
    } else {
        asc_vf_call<OpScatterCopySimtKernel<T, int64_t>>(dim3(THREADS<int64_t>), tilingData.totalElements,
                                                         tilingData.perCoreElements, xGm, yGm);
        GlobalTensor<T> yFlushGm;
        yFlushGm.SetGlobalBuffer(yGm);
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yFlushGm);
        asc_vf_call<OpScatterAddSimtKernel<T, IndexT, int64_t>>(
            dim3(THREADS<int64_t>), tilingData.K, tilingData.totalElements, tilingData.totalScatters,
            tilingData.updateDataNum, tilingData.perCoreElements, stridesUb, indicesGm, updatesGm, yGm);
    }
}

template <typename T, typename IndexT>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace,
                               const ScatterNonAliasingAddTilingData& tilingData)
{
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    constexpr uint32_t STRIDES_UB_WORDS = MAX_STRIDES * sizeof(int64_t) / sizeof(uint32_t);
    auto stridesRaw = ubAlloc.Alloc<uint32_t>(STRIDES_UB_WORDS);
    __ubuf__ int64_t* stridesUb = (__ubuf__ int64_t*)stridesRaw.GetPhyAddr();
    for (int32_t k = 0; k < MAX_STRIDES; k++) {
        stridesUb[k] = tilingData.strides[k];
    }
    DataSyncBarrier<MemDsbT::UB>();
    __gm__ T* xGm = (__gm__ T*)x;
    __gm__ IndexT* indicesGm = (__gm__ IndexT*)indices;
    __gm__ T* updatesGm = (__gm__ T*)updates;
    __gm__ T* yGm = (__gm__ T*)y;
    ScatterDispatch<T, IndexT>(tilingData, stridesUb, xGm, indicesGm, updatesGm, yGm);
}

} // namespace NsScatterNonAliasingAdd

#endif
