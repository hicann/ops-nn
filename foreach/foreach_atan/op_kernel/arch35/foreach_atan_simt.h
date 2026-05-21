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
 * \file foreach_atan_simt.h
 * \brief SIMT kernel implementation for foreach_atan
 */

#ifndef FOREACH_ATAN_SIMT_H
#define FOREACH_ATAN_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_atan_tiling_data.h"
#include "foreach_atan_tiling_key.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachAtan {

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
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

/**
 * \brief SIMT VF kernel: atan for float32 tensors (no type promotion needed)
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachAtanSimtFloat(
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

        __gm__ float* xData = SimtGetTensorAddr<float>(xList, t);
        __gm__ float* yData = SimtGetTensorAddr<float>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(
            AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            yData[idx] = atanf(xData[idx]);
        }
    }
}

/**
 * \brief SIMT VF kernel: atan for float16 tensors (type promotion: half -> float -> atanf -> half)
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachAtanSimtHalf(
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

        __gm__ half* xData = SimtGetTensorAddr<half>(xList, t);
        __gm__ half* yData = SimtGetTensorAddr<half>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(
            AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            float xVal = __half2float(xData[idx]);
            float yVal = atanf(xVal);
            yData[idx] = __float2half(yVal);
        }
    }
}

/**
 * \brief SIMT VF kernel: atan for bfloat16 tensors (type promotion: bf16 -> float -> atanf -> bf16)
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachAtanSimtBfloat16(
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

        __gm__ bfloat16_t* xData = SimtGetTensorAddr<bfloat16_t>(xList, t);
        __gm__ bfloat16_t* yData = SimtGetTensorAddr<bfloat16_t>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(
            AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            float xVal = __bfloat162float(xData[idx]);
            float yVal = atanf(xVal);
            yData[idx] = __float2bfloat16(yVal);
        }
    }
}

/**
 * \brief Process entry: launch SIMT VF for foreach_atan
 */
template <typename T>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR y,
    const __gm__ ForeachAtanTilingData* tilingGm)
{
    // Extract tensorElements array pointer from GM tiling data
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachAtanTilingData*>(tilingGm)) +
        offsetof(ForeachAtanTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    if constexpr (std::is_same_v<T, float>) {
        AscendC::Simt::VF_CALL<OpForeachAtanSimtFloat>(
            AscendC::Simt::Dim3(THREAD_NUM),
            tensorCount,
            elemCounts,
            x,
            y);
    } else if constexpr (std::is_same_v<T, half>) {
        AscendC::Simt::VF_CALL<OpForeachAtanSimtHalf>(
            AscendC::Simt::Dim3(THREAD_NUM),
            tensorCount,
            elemCounts,
            x,
            y);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        AscendC::Simt::VF_CALL<OpForeachAtanSimtBfloat16>(
            AscendC::Simt::Dim3(THREAD_NUM),
            tensorCount,
            elemCounts,
            x,
            y);
    }
}

} // namespace NsForeachAtan

#endif // FOREACH_ATAN_SIMT_H