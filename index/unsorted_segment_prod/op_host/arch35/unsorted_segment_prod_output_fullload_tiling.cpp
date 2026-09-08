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
 * \file unsorted_segment_prod_output_fullload_tiling.cpp
 * \brief unsorted_segment_prod_output_fullload_tiling
 */

#include "unsorted_segment_prod_output_fullload_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

bool UnsortedSegmentProdOutFlTiling::IsCapable()
{
    if (!UnsortedSegmentOutFlTiling::IsCapable()) {
        return false;
    }
    if (dataShapeSize_ == 0UL) {
        return true;
    }
    constexpr uint64_t OUTFL_ECONOMIC_FACTOR = 2;
    uint64_t outputSize = outputOuterDim_ * innerDim_;
    return inputOuterDim_ > outputSize * totalCoreNum_ * OUTFL_ECONOMIC_FACTOR;
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdOutFlTiling, 10);

} // namespace optiling
