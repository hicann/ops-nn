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
 * \file foreach_round_off_number_simt.h
 * \brief SIMT kernel implementation for foreach_round_off_number operator.
 *        Applies the requested element-wise rounding mode to a tensor list.
 */

#ifndef FOREACH_ROUND_OFF_NUMBER_SIMT_H
#define FOREACH_ROUND_OFF_NUMBER_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_round_off_number_tiling_data.h"
#include "foreach_round_off_number_tiling_key.h"

namespace NsForeachRoundOffNumber {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

constexpr int8_t CAST_NONE = 0;
constexpr int8_t CAST_RINT = 1;
constexpr int8_t CAST_FLOOR = 2;
constexpr int8_t CAST_CEIL = 3;
constexpr int8_t CAST_ROUND = 4;
constexpr int8_t CAST_TRUNC = 5;
constexpr int8_t CAST_ODD = 6;
constexpr int8_t CAST_FRAC = 7;

// ========== Helpers: apply the round mode with SIMT C APIs ==========

__simt_callee__ inline float RoundToOdd(float value)
{
    float floorValue = floorf(value);
    float fraction = value - floorValue;
    if (fraction > 0.5f) {
        return floorValue + 1.0f;
    }
    if (fraction < 0.5f) {
        return floorValue;
    }
    return fmodf(floorValue, 2.0f) != 0.0f ? floorValue : floorValue + 1.0f;
}

__simt_callee__ inline float ApplyRoundMode(float value, int8_t roundModeValue)
{
    switch (roundModeValue) {
        case CAST_RINT:
            return rintf(value);
        case CAST_FLOOR:
            return floorf(value);
        case CAST_CEIL:
            return ceilf(value);
        case CAST_ROUND:
            return roundf(value);
        case CAST_TRUNC:
            return truncf(value);
        case CAST_ODD:
            return RoundToOdd(value);
        case CAST_FRAC:
            return value - truncf(value);
        case CAST_NONE:
        default:
            return value;
    }
}

template <typename T>
__simt_callee__ inline float ConvertToFloat(T value)
{
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(value);
    } else {
        return __bfloat162float(value);
    }
}

template <typename T>
__simt_callee__ inline T ConvertFromFloat(float value)
{
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half_rn(value);
    } else {
        return __float2bfloat16_rn(value);
    }
}

// ========== Helper: get tensor address from ListTensorDesc ==========

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

// ========== SIMT VF kernel: grid-stride round on tensor list ==========

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachRoundOffNumberSimt(int32_t tensorId, int64_t count,
                                                                                        GM_ADDR xList, GM_ADDR yList,
                                                                                        __gm__ int8_t* roundMode)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    int8_t roundModeValue = *roundMode;
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        T xVal = xData[idx];
        if constexpr (std::is_same_v<T, int16_t>) {
            yData[idx] = roundModeValue == CAST_FRAC ? static_cast<int16_t>(0) : xVal;
        } else {
            yData[idx] = ConvertFromFloat<T>(ApplyRoundMode(ConvertToFloat(xVal), roundModeValue));
        }
    }
}

// ========== Process entry function ==========

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR roundMode, GM_ADDR y,
                               const ForeachRoundOffNumberTilingData* tilingData)
{
    for (int32_t tensorId = 0; tensorId < tilingData->tensorCount; tensorId++) {
        int64_t count = tilingData->tensorElements[tensorId];
        if (count <= 0) {
            continue;
        }
        AscendC::Simt::VF_CALL<OpForeachRoundOffNumberSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), tensorId, count, x, y,
                                                               reinterpret_cast<__gm__ int8_t*>(roundMode));
    }
}

} // namespace NsForeachRoundOffNumber

#endif // FOREACH_ROUND_OFF_NUMBER_SIMT_H
