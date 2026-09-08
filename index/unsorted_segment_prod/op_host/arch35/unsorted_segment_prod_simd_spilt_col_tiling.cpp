/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_prod_simd_spilt_col_tiling.cpp
 * \brief unsorted_segment_prod_simd_spilt_col_tiling
 */

#include "unsorted_segment_prod_simd_spilt_col_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

bool UnsortedSegmentProdSimdSplitColTiling::IsCapable()
{
    constexpr uint64_t LAST_DIM_SIMD_COND = 0;
    constexpr uint64_t BUFFER_NUM = 2;
    constexpr uint64_t PROD_BASE_A_SIZE = 224;
    constexpr uint64_t RATIO_BY_SORT = 500;
    constexpr uint64_t MIN_SPLIT_CORES = 4;
    constexpr uint64_t MIN_SPLIT_EFFECT_RATIO = 100;
    if (innerDim_ * dataTypeBytes_ > totalCoreNum_ * LAST_DIM_SIMD_COND && ratio_ < RATIO_BY_SORT) {
        normBlockData_ = Ops::Base::CeilAlign(Ops::Base::CeilDiv(innerDim_, totalCoreNum_),
                                              ubBlockSize_ / dataTypeBytes_);
        normBlockData_ = std::max(normBlockData_, LAST_DIM_SIMD_COND / dataTypeBytes_);
        usedCoreNum_ = Ops::Base::CeilDiv(innerDim_, normBlockData_);
        if (usedCoreNum_ < MIN_SPLIT_CORES && ratio_ > MIN_SPLIT_EFFECT_RATIO) {
            return false;
        }
        tailBlockData_ = innerDim_ - (usedCoreNum_ - 1UL) * normBlockData_;
        baseA_ = std::min(PROD_BASE_A_SIZE / dataTypeBytes_, normBlockData_);
        baseS_ = 1UL;
        outUbsize_ = outputOuterDim_ * baseA_ * dataTypeBytes_;
        uint64_t needUbSize = outUbsize_ + baseS_ * baseA_ * dataTypeBytes_ * BUFFER_NUM +
                              (baseS_ * idTypeBytes_ + ubBlockSize_) * BUFFER_NUM;
        return needUbSize < ubSize_;
    }
    return false;
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdSimdSplitColTiling, 20);

} // namespace optiling
