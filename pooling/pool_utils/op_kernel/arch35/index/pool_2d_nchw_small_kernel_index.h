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
 * \file pool_2d_nchw_small_kernel_index.h
 * \brief AvgPool/MaxPoolV3 NCHW small kernel 共用的 gather 索引生成入口，按 gather 模式分派到对应索引实现。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_2D_NCHW_SMALL_KERNEL_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_2D_NCHW_SMALL_KERNEL_INDEX_H_

#include <cstdint>

#include "kernel_operator.h"
#include "pool_utils/arch35/index/pool_2d_gather_scatter_index.h"

namespace PoolUtils {
namespace Index {

// gather 模式取值与 AvgPool/MaxPoolV3 算子侧 tiling 下发的 gatherMode 保持一致：
// GATHER_SINGLE_ROW = 0, GATHER_MULTI_ROW = 1, GATHER_MULTI_BATCH = 2, GATHER_SINGLE_KERNEL = 3。
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void GenGatherIndex(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElements, uint32_t wIn,
                                      uint32_t hStride, uint32_t wStride, int64_t kW, int64_t kH,
                                      AscendC::LocalTensor<U>& indexLocal)
{
    if constexpr (GATHER_MODE == 0) { // GATHER_SINGLE_ROW
        PoolUtils::Index::GenGatherIndexSingleRow<U>(wStride, indexLocal);
    } else if constexpr (GATHER_MODE == 1) { // GATHER_MULTI_ROW
        PoolUtils::Index::GenGatherIndexMultiRow<U>(wFactorOut, wIn, hStride, wStride, indexLocal);
    } else if constexpr (GATHER_MODE == 2) { // GATHER_MULTI_BATCH
        PoolUtils::Index::GenGatherIndexMultiBatch<U>(hFactorOut, wFactorOut, batchElements, wIn, hStride, wStride,
                                                      indexLocal);
    } else { // GATHER_SINGLE_KERNEL
        PoolUtils::Index::GenGatherIndexSingleKernel(wIn, kW, kH, indexLocal);
    }
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_2D_NCHW_SMALL_KERNEL_INDEX_H_
