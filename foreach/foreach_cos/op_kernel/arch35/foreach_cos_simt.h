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
 * \file foreach_cos_simt.h
 * \brief SIMT kernel implementation for foreach_cos
 */

#ifndef FOREACH_COS_SIMT_H
#define FOREACH_COS_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_cos_tiling_data.h"
#include "foreach_cos_tiling_key.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachCos {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

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
 * \brief SIMT VF kernel: compute cosine of all elements across all tensors
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachCosSimt(
    int32_t tensorCount,
    __gm__ int64_t* tensorElements,
    GM_ADDR xList,
    GM_ADDR yList)
{
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
            T xVal = xData[idx];
            float xF32 = ToFloat32<T>(xVal);
            float yF32 = cosf(xF32);
            yData[idx] = FromFloat32<T>(yF32);
        }
    }
}

/**
 * \brief Process entry: launch SIMT VF for foreach_cos
 */
template <typename T>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR y,
    const __gm__ ForeachCosTilingData* tilingGm)
{
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachCosTilingData*>(tilingGm)) +
        offsetof(ForeachCosTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    AscendC::Simt::VF_CALL<OpForeachCosSimt<T>>(
        AscendC::Simt::Dim3(THREAD_NUM),
        tensorCount,
        elemCounts,
        x,
        y);
}

} // namespace NsForeachCos

#endif // FOREACH_COS_SIMT_H
