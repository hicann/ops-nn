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
 * \file foreach_pow_scalar_list_simt.h
 * \brief SIMT kernel implementation for foreach_pow_scalar_list
 */

#ifndef FOREACH_POW_SCALAR_LIST_SIMT_H
#define FOREACH_POW_SCALAR_LIST_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_pow_scalar_list_tiling_data.h"
#include "foreach_pow_scalar_list_tiling_key.h"

namespace NsForeachPowScalarList {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_64 = 512;

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

template <typename T>
struct ComputeType {
    using type = T;
};
template <>
struct ComputeType<half> {
    using type = float;
};
template <>
struct ComputeType<bfloat16_t> {
    using type = float;
};
template <>
struct ComputeType<int32_t> {
    using type = int64_t;
};

template <typename T>
__simt_callee__ inline float ConvertToFloat(T val);

template <>
__simt_callee__ inline float ConvertToFloat<float>(float val)
{
    return val;
}

template <>
__simt_callee__ inline float ConvertToFloat<half>(half val)
{
    return __half2float(val);
}

template <>
__simt_callee__ inline float ConvertToFloat<bfloat16_t>(bfloat16_t val)
{
    return __bfloat162float(val);
}

template <typename T>
__simt_callee__ inline T ConvertFromFloat(float val);

template <>
__simt_callee__ inline float ConvertFromFloat<float>(float val)
{
    return val;
}

template <>
__simt_callee__ inline half ConvertFromFloat<half>(float val)
{
    return __float2half(val);
}

template <>
__simt_callee__ inline bfloat16_t ConvertFromFloat<bfloat16_t>(float val)
{
    return __float2bfloat16(val);
}

__simt_callee__ inline int64_t SimtPowInt(int64_t base, int64_t exp)
{
    if (exp < 0) {
        if (base == 1)
            return 1;
        if (base == -1)
            return (exp % 2 == 0) ? 1 : -1;
        return 0;
    }
    int64_t result = 1;
    int64_t b = base;
    uint64_t e = static_cast<uint64_t>(exp);
    while (e > 0) {
        if (e & 1) {
            result *= b;
        }
        b *= b;
        e >>= 1;
    }
    return result;
}

template <typename T, typename S>
__simt_callee__ inline void PowComputeBody(__gm__ T* xData, __gm__ T* yData, S scalarVal, uint64_t idx)
{
    using C = typename ComputeType<T>::type;
    if constexpr (std::is_same_v<C, int64_t>) {
        C xVal = static_cast<C>(xData[idx]);
        C sVal = static_cast<C>(scalarVal);
        yData[idx] = static_cast<T>(SimtPowInt(xVal, sVal));
    } else {
        float bF = ConvertToFloat<T>(xData[idx]);
        float eF = static_cast<float>(scalarVal);
        yData[idx] = ConvertFromFloat<T>(powf(bF, eF));
    }
}

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachPowScalarListSimt32(int32_t tensorId,
                                                                                         int64_t count, GM_ADDR xList,
                                                                                         GM_ADDR scalars, GM_ADDR yList)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
    S scalarVal = scalarsGm[tensorId];
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(blockDim.x * gridDim.x);
    for (uint32_t idx = tid; idx < static_cast<uint32_t>(count); idx += stride) {
        PowComputeBody<T, S>(xData, yData, scalarVal, static_cast<uint64_t>(idx));
    }
}

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_64) inline void OpForeachPowScalarListSimt64(
    int32_t tensorId, int64_t count, GM_ADDR xList, GM_ADDR scalars, GM_ADDR yList)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
    S scalarVal = scalarsGm[tensorId];
    uint64_t tid = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t stride = static_cast<uint64_t>(blockDim.x * gridDim.x);
    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        PowComputeBody<T, S>(xData, yData, scalarVal, idx);
    }
}

template <typename T, typename S>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR scalars, GM_ADDR y, const ForeachPowScalarListTilingData* tilingGm)
{
    constexpr int64_t kUint32Max = 0xFFFFFFFFLL;
    for (int32_t tensorId = 0; tensorId < tilingGm->tensorCount; tensorId++) {
        int64_t count = tilingGm->tensorElements[tensorId];
        if (count <= 0) {
            continue;
        }
        if (count <= kUint32Max) {
            asc_vf_call<OpForeachPowScalarListSimt32<T, S>>(dim3(THREAD_NUM), tensorId, count, x, scalars, y);
        } else {
            asc_vf_call<OpForeachPowScalarListSimt64<T, S>>(dim3(THREAD_NUM_64), tensorId, count, x, scalars, y);
        }
    }
}

} // namespace NsForeachPowScalarList

#endif // FOREACH_POW_SCALAR_LIST_SIMT_H
