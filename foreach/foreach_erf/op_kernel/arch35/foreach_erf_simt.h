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
#include "simt_api/math_functions.h"
#include "foreach_erf_tiling_data.h"
#include "foreach_erf_tiling_key.h"

namespace NsForeachErf {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

/**
 * \brief Compute erf via simt library erff (direct call, full-domain library
 *        semantics: NaN/±Inf/saturation and all special values are carried
 *        by erff itself, with no input-range pre-check or clamping)
 */
template <typename T>
__simt_callee__ inline T ComputeErf(T x)
{
    if constexpr (std::is_same_v<T, float>) {
        // float32: direct library call
        return erff(x);
    } else {
        // float16/bfloat16: promote to float32, compute via erff, then cast back
        float xFloat = static_cast<float>(x);
        float resultFloat = erff(xFloat);
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
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachErfSimt(int32_t tensorId, int64_t count,
                                                                             GM_ADDR xList, GM_ADDR yList)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        T xVal = xData[idx];
        T yVal = ComputeErf<T>(xVal);
        yData[idx] = yVal;
    }
}

/**
 * \brief Process entry: launch SIMT VF for foreach_erf
 */
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const ForeachErfTilingData* tilingGm)
{
    for (int32_t tensorId = 0; tensorId < tilingGm->tensorCount; tensorId++) {
        int64_t count = tilingGm->tensorElements[tensorId];
        if (count <= 0) {
            continue;
        }
        AscendC::Simt::VF_CALL<OpForeachErfSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), tensorId, count, x, y);
    }
}

} // namespace NsForeachErf

#endif // FOREACH_ERF_SIMT_H
