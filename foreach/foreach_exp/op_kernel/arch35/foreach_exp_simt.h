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
 * \file foreach_exp_simt.h
 * \brief SIMT kernel implementation for foreach_exp
 */

#ifndef FOREACH_EXP_SIMT_H
#define FOREACH_EXP_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_exp_tiling_data.h"
#include "foreach_exp_tiling_key.h"

namespace NsForeachExp {

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
 * \brief Promote half/bf16 to float32 for precision
 */
template <typename T>
__simt_callee__ inline float PromoteToFloat(T val);

template <>
__simt_callee__ inline float PromoteToFloat<half>(half val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<bfloat16_t>(bfloat16_t val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<float>(float val)
{
    return val;
}

/**
 * \brief Cast float32 back to half/bf16
 */
template <typename T>
__simt_callee__ inline T CastFromFloat(float val);

template <>
__simt_callee__ inline half CastFromFloat<half>(float val)
{
    return static_cast<half>(val);
}

template <>
__simt_callee__ inline bfloat16_t CastFromFloat<bfloat16_t>(float val)
{
    return static_cast<bfloat16_t>(val);
}

template <>
__simt_callee__ inline float CastFromFloat<float>(float val)
{
    return val;
}

/**
 * \brief SIMT VF kernel: compute exp for all elements across all tensors
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachExpSimt(int32_t tensorId, int64_t count,
                                                                             GM_ADDR xList, GM_ADDR yList)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        T xVal = xData[idx];
        float xFloat = PromoteToFloat<T>(xVal);
        float yFloat = expf(xFloat);
        yData[idx] = CastFromFloat<T>(yFloat);
    }
}

/**
 * \brief Process entry: launch SIMT VF for foreach_exp
 */
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const ForeachExpTilingData* tilingGm)
{
    for (int32_t tensorId = 0; tensorId < tilingGm->tensorCount; tensorId++) {
        int64_t count = tilingGm->tensorElements[tensorId];
        if (count <= 0) {
            continue;
        }
        AscendC::Simt::VF_CALL<OpForeachExpSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), tensorId, count, x, y);
    }
}

} // namespace NsForeachExp

#endif // FOREACH_EXP_SIMT_H
