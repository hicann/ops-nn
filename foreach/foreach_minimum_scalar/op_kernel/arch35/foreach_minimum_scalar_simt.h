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
 * \file foreach_minimum_scalar_simt.h
 * \brief SIMT kernel implementation for foreach_minimum_scalar
 */

#ifndef FOREACH_MINIMUM_SCALAR_SIMT_H
#define FOREACH_MINIMUM_SCALAR_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/math_functions.h"
#include "foreach_minimum_scalar_tiling_data.h"
#include "foreach_minimum_scalar_tiling_key.h"

namespace NsForeachMinimumScalar {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

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
 * \brief Type promotion helper: low-precision types promote to float32 for computation
 */
template <typename T>
struct ComputeType { using type = T; };
template <> struct ComputeType<half> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };

/**
 * \brief NaN-propagating minimum for float type (IEEE 754-2019 minimum)
 * NaN propagation: min(NaN, x) = NaN, min(x, NaN) = NaN
 * Zero comparison: min(+0.0, -0.0) = -0.0
 */
__simt_callee__ inline float SimtMinimum(float a, float b)
{
    // 1. NaN propagation: return NaN if either input is NaN
    if (isnan(a)) {
        return a;
    }
    if (isnan(b)) {
        return b;
    }

    // 2. Normal comparison: a < b returns a, b < a returns b
    if (a < b) {
        return a;
    }
    if (b < a) {
        return b;
    }

    // 3. a == b (including +0.0 == -0.0 case)
    // IEEE 754-2019 minimum: prefer -0.0 when equal
    // signbit returns non-zero for negative (including -0.0), zero for positive (including +0.0)
    if (signbit(a)) {
        return a;
    }
    return b;
}

/**
 * \brief Simple minimum for int32_t type (no NaN concept)
 */
__simt_callee__ inline int32_t SimtMinimum(int32_t a, int32_t b)
{
    return (a <= b) ? a : b;
}

/**
 * \brief SIMT VF kernel: compute min(x, scalar) for all elements across all tensors
 */
template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachMinimumScalarSimt(
    int32_t tensorCount,
    __gm__ int64_t* tensorElements,
    GM_ADDR xList,
    GM_ADDR yList,
    S scalarVal)
{
    using C = typename ComputeType<T>::type;

    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(
            AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            C xVal = static_cast<C>(xData[idx]);
            C sVal = static_cast<C>(scalarVal);
            C result = SimtMinimum(xVal, sVal);
            yData[idx] = static_cast<T>(result);
        }
    }
}

/**
 * \brief Process entry: read scalar from GM, launch SIMT VF for foreach_minimum_scalar
 */
template <typename T, typename S>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    const __gm__ ForeachMinimumScalarTilingData* tilingGm)
{
    // Read scalar value from GM (single-element tensor)
    __gm__ S* scalarGm = reinterpret_cast<__gm__ S*>(scalar);
    S scalarVal = *scalarGm;

    // Extract tensorElements array pointer from GM tiling data
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachMinimumScalarTilingData*>(tilingGm)) +
        offsetof(ForeachMinimumScalarTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    AscendC::Simt::VF_CALL<OpForeachMinimumScalarSimt<T, S>>(
        AscendC::Simt::Dim3(THREAD_NUM),
        tensorCount,
        elemCounts,
        x,
        y,
        scalarVal);
}

} // namespace NsForeachMinimumScalar

#endif // FOREACH_MINIMUM_SCALAR_SIMT_H