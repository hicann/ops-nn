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
 * \file foreach_asin_simt.h
 * \brief SIMT kernel implementation for foreach_asin
 */

#ifndef FOREACH_ASIN_SIMT_H
#define FOREACH_ASIN_SIMT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_asin_tiling_data.h"
#include "foreach_asin_tiling_key.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachAsin {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

/**
 * @brief 对单个元素计算 asin，支持类型提升
 * @tparam T 数据类型（float/half/bfloat16_t）
 */
template <typename T>
__simt_callee__ inline T ComputeAsin(T val)
{
    if constexpr (std::is_same<T, float>::value) {
        return asinf(val);
    } else if constexpr (std::is_same<T, half>::value) {
        float fVal = __half2float(val);
        float result = asinf(fVal);
        return __float2half(result);
    } else {
        // bfloat16_t
        float fVal = __bfloat162float(val);
        float result = asinf(fVal);
        return __float2bfloat16(result);
    }
}

/**
 * @brief SIMT VF kernel：对一个 tensor 的一段连续元素计算 asin
 * @tparam T 数据类型
 * @param localStart 该段在 tensor 内的起始索引
 * @param localEnd   该段在 tensor 内的结束索引（不包含）
 * @param xAddr      输入 tensor 的 GM 地址
 * @param yAddr      输出 tensor 的 GM 地址
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void ForeachAsinSimtKernel(
    int64_t localStart, int64_t localEnd,
    __gm__ T* xAddr, __gm__ T* yAddr)
{
    for (int64_t i = localStart + static_cast<int64_t>(Simt::GetThreadIdx());
         i < localEnd;
         i += static_cast<int64_t>(Simt::GetThreadNum())) {
        T val = xAddr[i];
        yAddr[i] = ComputeAsin<T>(val);
    }
}

/**
 * @brief Process 函数：按核切分处理所有 tensor
 * @tparam T 数据类型
 * @param x         输入 tensor list 的 GM 地址
 * @param y         输出 tensor list 的 GM 地址
 * @param tilingData tiling 数据指针
 */
template <typename T>
__aicore__ inline void Process(
    GM_ADDR x, GM_ADDR y,
    const ForeachAsinTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t coreStart = blockIdx * tilingData->perCoreElements;
    int64_t coreEnd = coreStart + tilingData->perCoreElements;

    if (coreEnd > tilingData->totalElements) {
        coreEnd = tilingData->totalElements;
    }
    // 空核直接返回
    if (coreStart >= tilingData->totalElements) {
        return;
    }

    // 使用 ListTensorDesc 解析各 tensor 的地址
    ListTensorDesc xList(reinterpret_cast<__gm__ void*>(x));
    ListTensorDesc yList(reinterpret_cast<__gm__ void*>(y));

    // 遍历所有 tensor，找出与当前核范围有交集的 tensor 并处理
    for (int32_t t = 0; t < tilingData->tensorNum; t++) {
        int64_t tensorStart = tilingData->tensorCumulativeOffset[t];
        int64_t tensorEnd = tilingData->tensorCumulativeOffset[t + 1];

        // 计算 [coreStart, coreEnd) 与 [tensorStart, tensorEnd) 的交集
        int64_t segStart = (coreStart > tensorStart) ? coreStart : tensorStart;
        int64_t segEnd = (coreEnd < tensorEnd) ? coreEnd : tensorEnd;

        if (segStart >= segEnd) {
            continue;
        }

        // 转换为 tensor 内的局部索引
        int64_t localStart = segStart - tensorStart;
        int64_t localEnd = segEnd - tensorStart;

        // 获取该 tensor 的 GM 地址
        __gm__ T* xAddr = xList.GetDataPtr<T>(t);
        __gm__ T* yAddr = yList.GetDataPtr<T>(t);

        // 启动 SIMT VF 计算该段
        Simt::VF_CALL<ForeachAsinSimtKernel<T>>(
            Simt::Dim3(THREAD_NUM),
            localStart, localEnd, xAddr, yAddr);
    }
}

} // namespace NsForeachAsin

#endif // FOREACH_ASIN_SIMT_H
