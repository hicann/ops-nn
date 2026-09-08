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
 * \file unsorted_segment_prod_sort_simt_tiling.cpp
 * \brief unsorted_segment_prod_sort_simt_tiling
 */

#include "unsorted_segment_prod_sort_simt_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

bool UnsortedSegmentProdSortSimtTiling::IsCapable()
{
    constexpr uint32_t INNER_DIM_ELEM_THRESHOLD = 32;
    constexpr uint32_t IN_OUT_RATE_THRESHOLD = 5;
    if (inputOuterDim_ / outputOuterDim_ >= IN_OUT_RATE_THRESHOLD && innerDim_ < INNER_DIM_ELEM_THRESHOLD) {
        return true;
    }
    return false;
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdSortSimtTiling, 60);

} // namespace optiling
