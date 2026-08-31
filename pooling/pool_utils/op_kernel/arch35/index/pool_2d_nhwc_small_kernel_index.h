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
 * \file pool_2d_nhwc_small_kernel_index.h
 * \brief AvgPool/MaxPoolV3 NHWC small kernel 共用的 gather 索引生成入口，按 gather 模式分派到对应索引实现。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_2D_NHWC_SMALL_KERNEL_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_2D_NHWC_SMALL_KERNEL_INDEX_H_

#include <cstdint>

#include "kernel_operator.h"
#include "pool_utils/arch35/index/pool_2d_gather_scatter_index.h"

namespace PoolUtils {
namespace Index {

// gather 模式取值，与 AvgPool/MaxPoolV3 算子侧 tiling 下发的 gatherMode 取值保持一致
constexpr int32_t GATHER_SINGLE_ROW = 0;
constexpr int32_t GATHER_MULTI_ROW = 1;

template <typename U, int32_t GATHER_MODE>
__aicore__ inline void GenGatherIndex(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn, uint32_t wInElms,
                                      uint32_t hStride, uint32_t wStride, uint32_t channels,
                                      AscendC::LocalTensor<U>& indexLocal)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        PoolUtils::Index::NHWCGenGatherIndexSingleRow<U>(wStride, channels, indexLocal);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        PoolUtils::Index::NHWCGenGatherIndexMultiRow<U>(wFactorOut, wInElms, hStride, wStride, channels, indexLocal);
    } else {
        PoolUtils::Index::NHWCGenGatherIndexMultiBatch<U>(hFactorOut, wFactorOut, hIn, wInElms, hStride, wStride,
                                                          channels, indexLocal);
    }
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_2D_NHWC_SMALL_KERNEL_INDEX_H_
