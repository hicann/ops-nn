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
 * \file foreach_tan_simt.h
 * \brief SIMT kernel implementation for foreach_tan operator
 */

#ifndef ASCENDC_FOREACH_TAN_SIMT_H
#define ASCENDC_FOREACH_TAN_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_tan_tiling_data.h"
#include "foreach_tan_tiling_key.h"

namespace NsForeachTan {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

// float32 版本 VF 函数：直接调用 tanf
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachTanSimt(
    __gm__ T* xPtr, __gm__ T* yPtr, int64_t elemCount)
{
    for (uint64_t idx = static_cast<uint64_t>(
             Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
         idx < static_cast<uint64_t>(elemCount);
         idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
        T val = xPtr[idx];
        yPtr[idx] = tanf(val);
    }
}

// float16 专用 VF 函数：half → float32 → tanf → half
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachTanHalfSimt(
    __gm__ half* xPtr, __gm__ half* yPtr, int64_t elemCount)
{
    for (uint64_t idx = static_cast<uint64_t>(
             Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
         idx < static_cast<uint64_t>(elemCount);
         idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
        half val = xPtr[idx];
        float val_f32 = __half2float(val);
        float result_f32 = tanf(val_f32);
        yPtr[idx] = __float2half(result_f32);
    }
}

// bfloat16 专用 VF 函数：bfloat16 → float32 → tanf → bfloat16
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void OpForeachTanBf16Simt(
    __gm__ bfloat16_t* xPtr, __gm__ bfloat16_t* yPtr, int64_t elemCount)
{
    for (uint64_t idx = static_cast<uint64_t>(
             Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
         idx < static_cast<uint64_t>(elemCount);
         idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
        bfloat16_t val = xPtr[idx];
        float val_f32 = __bfloat162float(val);
        float result_f32 = tanf(val_f32);
        yPtr[idx] = __float2bfloat16(result_f32);
    }
}

// Process 入口函数：逐 Tensor 调用 VF_CALL
template <typename T>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR y, const ForeachTanTilingData* tilingData)
{
    ListTensorDesc xList(reinterpret_cast<__gm__ void*>(x));
    ListTensorDesc yList(reinterpret_cast<__gm__ void*>(y));

    for (int32_t i = 0; i < tilingData->tensorNum; i++) {
        int64_t count = tilingData->tensorElements[i];
        if (count == 0) {
            continue;  // 空 Tensor 跳过
        }

        __gm__ T* xPtr = xList.GetDataPtr<T>(i);
        __gm__ T* yPtr = yList.GetDataPtr<T>(i);

        // 根据 T 类型选择对应的 VF 函数
        // float32: OpForeachTanSimt<float>
        // half: OpForeachTanHalfSimt
        // bfloat16_t: OpForeachTanBf16Simt
        if constexpr (std::is_same_v<T, float>) {
            Simt::VF_CALL<OpForeachTanSimt<float>>(
                Simt::Dim3(THREAD_NUM), xPtr, yPtr, count);
        } else if constexpr (std::is_same_v<T, half>) {
            Simt::VF_CALL<OpForeachTanHalfSimt>(
                Simt::Dim3(THREAD_NUM), xPtr, yPtr, count);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            Simt::VF_CALL<OpForeachTanBf16Simt>(
                Simt::Dim3(THREAD_NUM), xPtr, yPtr, count);
        }
    }
}

} // namespace NsForeachTan

#endif