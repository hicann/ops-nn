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
 * \file foreach_log_simt.h
 * \brief SIMT kernel implementation for foreach_log
 */

#ifndef FOREACH_LOG_SIMT_H
#define FOREACH_LOG_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_log_tiling_data.h"
#include "foreach_log_tiling_key.h"

namespace NsForeachLog {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

template <typename T>
__simt_callee__ inline T LogFunc(T x);

template <>
__simt_callee__ inline float LogFunc<float>(float x) {
    return logf(x);
}

template <>
__simt_callee__ inline half LogFunc<half>(half x) {
    return hlog(x);
}

template <>
__simt_callee__ inline bfloat16_t LogFunc<bfloat16_t>(bfloat16_t x) {
    return hlog(x);
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachLogSimt(
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
            blockIdx.x * blockDim.x +
            threadIdx.x);
        uint64_t stride = static_cast<uint64_t>(
            blockDim.x * gridDim.x);

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            T val = xData[idx];
            yData[idx] = LogFunc<T>(val);
        }
    }
}

template <typename T>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR y,
    const __gm__ ForeachLogTilingData* tilingGm)
{
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachLogTilingData*>(tilingGm)) +
        offsetof(ForeachLogTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    asc_vf_call<OpForeachLogSimt<T>>(
        dim3(THREAD_NUM),
        tensorCount,
        elemCounts,
        x,
        y);
}

} // namespace NsForeachLog

#endif // FOREACH_LOG_SIMT_H
