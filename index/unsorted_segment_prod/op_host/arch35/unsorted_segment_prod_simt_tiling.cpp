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
 * \file unsorted_segment_prod_simt_tiling.cpp
 * \brief unsorted_segment_prod_simt_tiling
 */

#include "unsorted_segment_prod_simt_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

ge::graphStatus UnsortedSegmentProdSimtTiling::DoOpTiling()
{
    uint64_t inputLength = dataShapeSize_;
    uint64_t outSize = outputOuterDim_ * innerDim_;
    constexpr uint64_t WARP = 32;
    constexpr uint64_t PROD_UB_MIN_FACTOR = 256;

    uint64_t curEff = Ops::Base::CeilDiv(inputLength, static_cast<uint64_t>(maxThread_));
    if (curEff < totalCoreNum_ / 2) {
        uint64_t dynMaxThread = Ops::Base::CeilDiv(inputLength, totalCoreNum_);
        dynMaxThread = Ops::Base::CeilAlign(dynMaxThread, WARP);
        dynMaxThread = std::max(dynMaxThread, WARP);
        dynMaxThread = std::min(dynMaxThread, static_cast<uint64_t>(maxThread_));
        maxThread_ = static_cast<uint32_t>(dynMaxThread);
    }
    usedCoreNum_ = Ops::Base::CeilDiv(inputLength, static_cast<uint64_t>(maxThread_));
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
    normBlockData = std::max(normBlockData, PROD_UB_MIN_FACTOR / dataTypeBytes_);
    uint64_t initUsedCore = Ops::Base::CeilDiv(outSize, normBlockData);
    usedCoreNum_ = std::max(usedCoreNum_, initUsedCore);
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdSimtTiling, 100);

} // namespace optiling
