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
 *        Computes element-wise round (half to even) on a tensor list.
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
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachRoundOffNumberSimt(int32_t tensorCount,
                                                                                        __gm__ int64_t* tensorElements,
                                                                                        GM_ADDR xList, GM_ADDR yList,
                                                                                        __gm__ int8_t* roundMode)
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
            if constexpr (std::is_same_v<T, float>) {
                yData[idx] = rintf(xVal);
            } else if constexpr (std::is_same_v<T, half>) {
                yData[idx] = hrint(xVal);
            } else {
                yData[idx] = static_cast<bfloat16_t>(rintf(static_cast<float>(xVal)));
            }
        }
    }
}

// ========== Process entry function ==========

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR roundMode, GM_ADDR y, GM_ADDR tiling)
{
    __gm__ const ForeachRoundOffNumberTilingData*
        tilingGm = reinterpret_cast<__gm__ const ForeachRoundOffNumberTilingData*>(tiling);

    __gm__ int64_t* elemCounts = const_cast<__gm__ int64_t*>(tilingGm->tensorElements);
    int32_t tensorCount = tilingGm->tensorCount;

    AscendC::Simt::VF_CALL<OpForeachRoundOffNumberSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), tensorCount, elemCounts, x,
                                                           y, reinterpret_cast<__gm__ int8_t*>(roundMode));
}

} // namespace NsForeachRoundOffNumber

#endif // FOREACH_ROUND_OFF_NUMBER_SIMT_H
