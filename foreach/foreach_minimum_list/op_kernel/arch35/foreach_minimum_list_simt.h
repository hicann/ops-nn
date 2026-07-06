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
 * \file foreach_minimum_list_simt.h
 * \brief SIMT kernel implementation for foreach_minimum_list
 */

#ifndef FOREACH_MINIMUM_LIST_SIMT_H
#define FOREACH_MINIMUM_LIST_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_minimum_list_tiling_data.h"
#include "foreach_minimum_list_tiling_key.h"
#include "kernel_operator_list_tensor_intf.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachMinimumList {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

// ===== float32 VF kernel (NaN-propagating minimum) =====
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachMinimumListSimtFp32(int64_t count, __gm__ T* x1,
                                                                                       __gm__ T* x2, __gm__ T* y)
{
    for (int64_t idx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx()); idx < count;
         idx += static_cast<int64_t>(AscendC::Simt::GetThreadNum())) {
        T a = x1[idx];
        T b = x2[idx];
        // NaN propagation minimum: (isnan(a) || a < b) ? a : b
        y[idx] = (isnan(static_cast<float>(a)) || a < b) ? a : b;
    }
}

// ===== float16 VF kernel (NaN-propagating minimum) =====
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachMinimumListSimtFp16(int64_t count, __gm__ half* x1,
                                                                                       __gm__ half* x2, __gm__ half* y)
{
    for (int64_t idx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx()); idx < count;
         idx += static_cast<int64_t>(AscendC::Simt::GetThreadNum())) {
        half a = x1[idx];
        half b = x2[idx];
        // NaN propagation minimum: (__hisnan(a) || a < b) ? a : b
        y[idx] = (__hisnan(a) || a < b) ? a : b;
    }
}

// ===== bfloat16 VF kernel (NaN-propagating minimum) =====
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachMinimumListSimtBf16(int64_t count,
                                                                                       __gm__ bfloat16_t* x1,
                                                                                       __gm__ bfloat16_t* x2,
                                                                                       __gm__ bfloat16_t* y)
{
    for (int64_t idx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx()); idx < count;
         idx += static_cast<int64_t>(AscendC::Simt::GetThreadNum())) {
        bfloat16_t a = x1[idx];
        bfloat16_t b = x2[idx];
        // bfloat16 -> float for NaN check and comparison
        float fa = __bfloat162float(a);
        float fb = __bfloat162float(b);
        y[idx] = (isnan(fa) || fa < fb) ? a : b;
    }
}

// ===== int32 VF kernel (no NaN, simple min) =====
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachMinimumListSimtInt32(int64_t count,
                                                                                        __gm__ int32_t* x1,
                                                                                        __gm__ int32_t* x2,
                                                                                        __gm__ int32_t* y)
{
    for (int64_t idx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx()); idx < count;
         idx += static_cast<int64_t>(AscendC::Simt::GetThreadNum())) {
        int32_t a = x1[idx];
        int32_t b = x2[idx];
        y[idx] = (a < b) ? a : b;
    }
}

// ===== Process: multi-core iterate tensor list =====
template <typename T, int32_t schMode>
__aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const ForeachMinimumListTilingData* tilingData)
{
    // 1. Compute core element range
    int64_t coreStart = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->perCoreElements;
    int64_t coreEnd = coreStart + tilingData->perCoreElements;
    if (static_cast<int32_t>(AscendC::GetBlockIdx()) == tilingData->needCoreNum - 1) {
        coreEnd = tilingData->totalElements;
    }
    if (coreStart >= coreEnd) {
        return;
    }

    // 2. Initialize ListTensorDesc
    ListTensorDesc x1List(reinterpret_cast<__gm__ void*>(x1));
    ListTensorDesc x2List(reinterpret_cast<__gm__ void*>(x2));
    ListTensorDesc yList(reinterpret_cast<__gm__ void*>(y));

    // 3. Iterate tensors, find overlap, compute
    int64_t tStart = 0;
    for (int32_t t = 0; t < tilingData->tensorNum; t++) {
        int64_t tSize = tilingData->tensorSizes[t];
        int64_t tEnd = tStart + tSize;

        int64_t oS = (coreStart > tStart) ? coreStart : tStart;
        int64_t oE = (coreEnd < tEnd) ? coreEnd : tEnd;

        if (oS < oE) {
            int64_t localOff = oS - tStart;
            int64_t cnt = oE - oS;
            __gm__ T* x1P = x1List.GetDataPtr<T>(t) + localOff;
            __gm__ T* x2P = x2List.GetDataPtr<T>(t) + localOff;
            __gm__ T* yP = yList.GetDataPtr<T>(t) + localOff;

            if constexpr (schMode == 0) {
                AscendC::Simt::VF_CALL<ForeachMinimumListSimtFp32<T>>(AscendC::Simt::Dim3(THREAD_NUM), cnt, x1P, x2P,
                                                                      yP);
            } else if constexpr (schMode == 1) {
                AscendC::Simt::VF_CALL<ForeachMinimumListSimtFp16>(
                    AscendC::Simt::Dim3(THREAD_NUM), cnt, reinterpret_cast<__gm__ half*>(x1P),
                    reinterpret_cast<__gm__ half*>(x2P), reinterpret_cast<__gm__ half*>(yP));
            } else if constexpr (schMode == 2) {
                AscendC::Simt::VF_CALL<ForeachMinimumListSimtBf16>(
                    AscendC::Simt::Dim3(THREAD_NUM), cnt, reinterpret_cast<__gm__ bfloat16_t*>(x1P),
                    reinterpret_cast<__gm__ bfloat16_t*>(x2P), reinterpret_cast<__gm__ bfloat16_t*>(yP));
            } else if constexpr (schMode == 3) {
                AscendC::Simt::VF_CALL<ForeachMinimumListSimtInt32>(
                    AscendC::Simt::Dim3(THREAD_NUM), cnt, reinterpret_cast<__gm__ int32_t*>(x1P),
                    reinterpret_cast<__gm__ int32_t*>(x2P), reinterpret_cast<__gm__ int32_t*>(yP));
            }
        }
        tStart = tEnd;
    }
}

} // namespace NsForeachMinimumList
#endif // FOREACH_MINIMUM_LIST_SIMT_H
