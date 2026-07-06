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
 * \file foreach_erf_simt.h
 * \brief SIMT kernel implementation for foreach_erf
 */

#ifndef FOREACH_ERF_SIMT_H
#define FOREACH_ERF_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_erf_tiling_data.h"
#include "foreach_erf_tiling_key.h"

namespace NsForeachErf {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
constexpr float ERF_CLAMP_MIN = -3.92f;
constexpr float ERF_CLAMP_MAX = 3.92f;

/**
 * \brief Clamp input value to specified range
 */
__simt_callee__ inline float Clamp(float x, float minVal, float maxVal)
{
    if (x < minVal) {
        return minVal;
    }
    if (x > maxVal) {
        return maxVal;
    }
    return x;
}

/**
 * \brief Erf approximation using rational function (v200+ platform)
 *        Polynomial coefficients from TBE erf implementation
 */
__simt_callee__ inline float ErfApproximation(float x)
{
    // 1. Clamp input to [-3.92, 3.92]
    float xClamped = Clamp(x, ERF_CLAMP_MIN, ERF_CLAMP_MAX);

    // 2. Compute x^2
    float x2 = xClamped * xClamped;

    // 3. Compute numerator polynomial
    // numer = ((((0.53443748819e-1*x² + 0.75517016694e1)*x² + 0.10162808918e3)*x²
    //           + 0.13938061484e4)*x² + 0.50637915060e4)*x² + 0.29639384698e5)*x
    float numer = x2;
    numer = numer * 0.53443748819e-1f + 0.75517016694e1f;
    numer = numer * x2 + 0.10162808918e3f;
    numer = numer * x2 + 0.13938061484e4f;
    numer = numer * x2 + 0.50637915060e4f;
    numer = numer * x2 + 0.29639384698e5f;
    numer = numer * xClamped;

    // 4. Compute denominator polynomial
    // denom = (((((x² + 0.31212858877e2)*x² + 0.39856963806e3)*x²
    //            + 0.30231248150e4)*x² + 0.13243365831e5)*x² + 0.26267224157e5
    float denom = x2;
    denom = denom + 0.31212858877e2f;
    denom = denom * x2 + 0.39856963806e3f;
    denom = denom * x2 + 0.30231248150e4f;
    denom = denom * x2 + 0.13243365831e5f;
    denom = denom * x2 + 0.26267224157e5f;

    // 5. Return erf result
    return numer / denom;
}

/**
 * \brief Handle special values (NaN and Inf) for erf
 */
__simt_callee__ inline float HandleSpecialValues(float x)
{
    // NaN detection: NaN != NaN
    if (x != x) {
        // Return NaN (using 0.0f / 0.0f to generate quiet NaN)
        return 0.0f / 0.0f;
    }
    // Inf detection: check large values beyond clamp range
    // For erf, x >= 3.92 outputs ~1, x <= -3.92 outputs ~-1
    // We use clamp to handle large values, so no explicit Inf check needed
    // But for true Inf, the result should be ±1
    // Since clamp will convert Inf to max/min, we handle it via approximation

    // Normal value -> use rational function approximation (includes clamp)
    return ErfApproximation(x);
}

/**
 * \brief Compute erf with type promotion for float16/bfloat16
 */
template <typename T>
__simt_callee__ inline T ComputeErf(T x)
{
    if constexpr (std::is_same_v<T, float>) {
        // float32: direct computation
        return HandleSpecialValues(x);
    } else {
        // float16/bfloat16: promote to float32, compute, then cast back
        float xFloat = static_cast<float>(x);
        float resultFloat = HandleSpecialValues(xFloat);
        return static_cast<T>(resultFloat);
    }
}

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
 * \brief SIMT VF kernel: apply erf to all elements across all tensors
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachErfSimt(int32_t tensorCount,
                                                                             __gm__ int64_t* tensorElements,
                                                                             GM_ADDR xList, GM_ADDR yList)
{
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                             AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            T xVal = xData[idx];
            T yVal = ComputeErf<T>(xVal);
            yData[idx] = yVal;
        }
    }
}

/**
 * \brief Process entry: launch SIMT VF for foreach_erf
 */
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const __gm__ ForeachErfTilingData* tilingGm)
{
    // Extract tensorElements array pointer from GM tiling data
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(const_cast<__gm__ ForeachErfTilingData*>(tilingGm)) +
        offsetof(ForeachErfTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    AscendC::Simt::VF_CALL<OpForeachErfSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), tensorCount, elemCounts, x, y);
}

} // namespace NsForeachErf

#endif // FOREACH_ERF_SIMT_H