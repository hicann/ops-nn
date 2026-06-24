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
constexpr float FLOAT_MANTISSA_INT_THRESHOLD = static_cast<float>(1 << 23);
constexpr float INT32_OVERFLOW_THRESHOLD = static_cast<float>(1LL << 31);

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

template <typename T>
struct ComputeType { using type = T; };
template <> struct ComputeType<half> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };
template <> struct ComputeType<int32_t> { using type = int64_t; };

static __simt_callee__ inline bool SimtIsNaN(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

static __simt_callee__ inline bool SimtIsPosInf(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return bits == 0x7F800000u;
}

static __simt_callee__ inline bool SimtIsNegInf(float v)
{
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return bits == 0xFF800000u;
}

static __simt_callee__ inline bool IsIntegerExp(float exp)
{
    if (fabsf(exp) >= FLOAT_MANTISSA_INT_THRESHOLD) return true;
    float rounded = rintf(exp);
    return fabsf(exp - rounded) < 1e-6f;
}

static __simt_callee__ inline bool IsOddInteger(float exp)
{
    if (fabsf(exp) >= INT32_OVERFLOW_THRESHOLD) {
        return fmodf(fabsf(exp), 2.0f) == 1.0f;
    }
    int32_t n = static_cast<int32_t>(rintf(exp));
    return (n & 1) != 0;
}

static __simt_callee__ inline float FloatToInf(bool positive)
{
    uint32_t bits = positive ? 0x7F800000u : 0xFF800000u;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

static __simt_callee__ inline float NegBaseIntExp(float base, float exp)
{
    float absBase = fabsf(base);
    if (fabsf(exp) >= INT32_OVERFLOW_THRESHOLD) {
        float result = powf(absBase, exp);
        float rem = fmodf(fabsf(exp), 2.0f);
        if (rem == 1.0f && base < 0.0f) {
            result = -result;
        }
        return result;
    }
    int32_t n = static_cast<int32_t>(rintf(exp));
    bool isNegExp = (n < 0);
    if (isNegExp) {
        n = -n;
    }
    float result = 1.0f;
    float cur = absBase;
    int32_t m = n;
    while (m > 0) {
        if (m & 1) {
            result *= cur;
        }
        cur *= cur;
        m >>= 1;
    }
    if (isNegExp) {
        result = 1.0f / result;
    }
    if ((n & 1) && base < 0.0f) {
        result = -result;
    }
    return result;
}

template <typename T>
__simt_callee__ inline float ConvertToFloat(T val);

template <>
__simt_callee__ inline float ConvertToFloat<float>(float val) { return val; }

template <>
__simt_callee__ inline float ConvertToFloat<half>(half val) { return __half2float(val); }

template <>
__simt_callee__ inline float ConvertToFloat<bfloat16_t>(bfloat16_t val) { return __bfloat162float(val); }

template <typename T>
__simt_callee__ inline T ConvertFromFloat(float val);

template <>
__simt_callee__ inline float ConvertFromFloat<float>(float val) { return val; }

template <>
__simt_callee__ inline half ConvertFromFloat<half>(float val) { return __float2half(val); }

template <>
__simt_callee__ inline bfloat16_t ConvertFromFloat<bfloat16_t>(float val) { return __float2bfloat16(val); }

__simt_callee__ inline int64_t SimtPowInt(int64_t base, int64_t exp)
{
    if (exp < 0) {
        if (base == 1) return 1;
        if (base == -1) return (exp % 2 == 0) ? 1 : -1;
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

static __simt_callee__ inline float SimtPowFloatSafe(float bF, float eF)
{
    if (SimtIsNaN(bF) || SimtIsNaN(eF)) {
        uint32_t nanBits = 0x7FC00000u;
        float result;
        __builtin_memcpy(&result, &nanBits, sizeof(result));
        return result;
    }
    if (eF == 0.0f) {
        return 1.0f;
    }
    if (bF == 1.0f) {
        return 1.0f;
    }
    if (SimtIsPosInf(bF)) {
        return (eF > 0.0f) ? FloatToInf(true) : 0.0f;
    }
    if (SimtIsNegInf(bF)) {
        if (SimtIsPosInf(eF)) {
            return FloatToInf(true);
        }
        if (SimtIsNegInf(eF)) {
            return 0.0f;
        }
        if (eF > 0.0f && IsIntegerExp(eF)) {
            return IsOddInteger(eF) ? FloatToInf(false) : FloatToInf(true);
        }
        if (eF < 0.0f && IsIntegerExp(eF)) {
            return IsOddInteger(-eF) ? -0.0f : 0.0f;
        }
        if (eF > 0.0f) {
            return FloatToInf(true);
        }
        return 0.0f;
    }
    if (SimtIsPosInf(eF)) {
        float absB = fabsf(bF);
        if (absB > 1.0f) return FloatToInf(true);
        if (absB < 1.0f) return 0.0f;
        if (bF == -1.0f) return 1.0f;
        return 0.0f;
    }
    if (SimtIsNegInf(eF)) {
        float absB = fabsf(bF);
        if (absB > 1.0f) return 0.0f;
        if (absB < 1.0f) return FloatToInf(true);
        if (bF == -1.0f) return 1.0f;
        return FloatToInf(true);
    }
    if (bF < 0.0f && IsIntegerExp(eF)) {
        return NegBaseIntExp(bF, eF);
    }
    return powf(bF, eF);
}

template <typename T, typename S>
__simt_callee__ inline void PowComputeBody(
    __gm__ T* xData, __gm__ T* yData, S scalarVal, uint64_t idx)
{
    using C = typename ComputeType<T>::type;
    if (static_cast<C>(xData[idx]) == static_cast<C>(1)) {
        yData[idx] = static_cast<T>(1);
        return;
    }
    if (scalarVal == 0) {
        yData[idx] = static_cast<T>(1);
        return;
    }
    if constexpr (std::is_same_v<C, int64_t>) {
        C xVal = static_cast<C>(xData[idx]);
        C sVal = static_cast<C>(scalarVal);
        yData[idx] = static_cast<T>(SimtPowInt(xVal, sVal));
    } else {
        float bF = ConvertToFloat<T>(xData[idx]);
        float eF = static_cast<float>(scalarVal);
        yData[idx] = ConvertFromFloat<T>(SimtPowFloatSafe(bF, eF));
    }
}

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachPowScalarListSimt32(
    int32_t tensorCount,
    __gm__ int64_t* tensorElements,
    GM_ADDR xList,
    GM_ADDR scalars,
    GM_ADDR yList)
{
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
        S scalarVal = scalarsGm[t];

        uint32_t tid = static_cast<uint32_t>(
            blockIdx.x * blockDim.x +
            threadIdx.x);
        uint32_t stride = static_cast<uint32_t>(
            blockDim.x * gridDim.x);

        for (uint32_t idx = tid; idx < static_cast<uint32_t>(count); idx += stride) {
            PowComputeBody<T, S>(xData, yData, scalarVal, static_cast<uint64_t>(idx));
        }
    }
}

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_64)
inline void OpForeachPowScalarListSimt64(
    int32_t tensorCount,
    __gm__ int64_t* tensorElements,
    GM_ADDR xList,
    GM_ADDR scalars,
    GM_ADDR yList)
{
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
        S scalarVal = scalarsGm[t];

        uint64_t tid = static_cast<uint64_t>(
            blockIdx.x * blockDim.x +
            threadIdx.x);
        uint64_t stride = static_cast<uint64_t>(
            blockDim.x * gridDim.x);

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            PowComputeBody<T, S>(xData, yData, scalarVal, idx);
        }
    }
}

template <typename T, typename S>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR scalars, GM_ADDR y,
    const __gm__ ForeachPowScalarListTilingData* tilingGm)
{
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachPowScalarListTilingData*>(tilingGm)) +
        offsetof(ForeachPowScalarListTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    constexpr int64_t kUint32Max = 0xFFFFFFFFLL;
    if (tilingGm->totalElements <= kUint32Max) {
        asc_vf_call<OpForeachPowScalarListSimt32<T, S>>(
            dim3(THREAD_NUM),
            tensorCount,
            elemCounts,
            x,
            scalars,
            y);
    } else {
        asc_vf_call<OpForeachPowScalarListSimt64<T, S>>(
            dim3(THREAD_NUM_64),
            tensorCount,
            elemCounts,
            x,
            scalars,
            y);
    }
}

} // namespace NsForeachPowScalarList

#endif // FOREACH_POW_SCALAR_LIST_SIMT_H
