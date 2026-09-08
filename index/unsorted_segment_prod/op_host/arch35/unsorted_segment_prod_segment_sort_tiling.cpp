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
 * \file unsorted_segment_prod_segment_sort_tiling.cpp
 * \brief unsorted_segment_prod_segment_sort_tiling
 */

#include "unsorted_segment_prod_segment_sort_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

namespace {
constexpr uint64_t SS_SYS_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint64_t SS_SORT_TILE = 8192UL;
constexpr uint64_t SS_INNER_MIN = 128UL;
constexpr uint64_t SS_INNER_MAX = 512UL;
constexpr uint64_t SS_MIN_INPUT_ROWS = 1024UL;
constexpr uint64_t SS_WS_ALIGN = 8UL;
constexpr uint64_t SS_WS_GUARD = 128UL;
constexpr uint64_t SS_WS_AREA_CNT = 4UL;
constexpr uint64_t SS_WS_ELEM_BYTES = 4UL;
} // namespace

bool UnsortedSegmentProdSegmentSortTiling::IsCapable()
{
    if (dataTypeBytes_ != 4UL) {
        return false;
    }
    if (dataShapeSize_ == 0UL || innerDim_ == 0UL || outputOuterDim_ == 0UL || inputOuterDim_ == 0UL) {
        return false;
    }
    if (innerDim_ < SS_INNER_MIN || innerDim_ >= SS_INNER_MAX) {
        return false;
    }
    if (inputOuterDim_ < SS_MIN_INPUT_ROWS) {
        return false;
    }
    uint64_t normBlock = Ops::Base::CeilAlign(Ops::Base::CeilDiv(innerDim_, totalCoreNum_),
                                              ubBlockSize_ / dataTypeBytes_);
    if (normBlock != ubBlockSize_ / dataTypeBytes_) {
        return false;
    }
    constexpr uint64_t INT32_MAX_BOUND = 2147483647UL;
    if (outputOuterDim_ > INT32_MAX_BOUND) {
        return false;
    }
    return true;
}

ge::graphStatus UnsortedSegmentProdSegmentSortTiling::DoOpTiling()
{
    uint64_t rowNum = inputOuterDim_;
    uint64_t blockNum = (rowNum == 0UL) ? 1UL : (rowNum < totalCoreNum_ ? rowNum : totalCoreNum_);
    uint64_t perCoreRows = (blockNum == 0UL) ? 0UL : Ops::Base::CeilDiv(rowNum, blockNum);
    blockNum = (perCoreRows == 0UL) ? 1UL : Ops::Base::CeilDiv(rowNum, perCoreRows);
    uint64_t tailCoreRows = rowNum - (blockNum - 1UL) * perCoreRows;

    blockNum_ = blockNum;
    blockTilingSize_ = perCoreRows * innerDim_;
    tailBlockTilingSize_ = tailCoreRows * innerDim_;
    usedCoreNum_ = blockNum;

    padM_ = rowNum;
    if (rowNum > SS_SORT_TILE) {
        uint64_t runCnt = 1UL;
        while (runCnt * 2UL <= blockNum) {
            runCnt *= 2UL;
        }
        while (Ops::Base::CeilDiv(rowNum, runCnt) > SS_SORT_TILE) {
            runCnt *= 2UL;
        }
        padM_ = runCnt * Ops::Base::CeilDiv(rowNum, runCnt);
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentProdSegmentSortTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentProdSegmentSortTilingData*
        tilingData = context_->GetTilingData<UnsortedSegment::UnsortedSegmentProdSegmentSortTilingData>();
    tilingData->blockNum = blockNum_;
    tilingData->blockTilingSize = blockTilingSize_;
    tilingData->tailBlockTilingSize = tailBlockTilingSize_;
    tilingData->innerDim = innerDim_;
    tilingData->outputOuterDim = outputOuterDim_;
}

uint64_t UnsortedSegmentProdSegmentSortTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_SEGMENT_SORT, USS_CAST_NONE);
    return tilingKey;
}

ge::graphStatus UnsortedSegmentProdSegmentSortTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    uint64_t sliceAlign = Ops::Base::CeilAlign(innerDim_, SS_WS_ALIGN);
    uint64_t mAlign = Ops::Base::CeilAlign(padM_, SS_WS_ALIGN) + SS_WS_GUARD;
    workspaces[0] = SS_SYS_WORKSPACE + mAlign * SS_WS_AREA_CNT * SS_WS_ELEM_BYTES +
                    blockNum_ * sliceAlign * SS_WS_ELEM_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentProdSegmentSortTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentProdSegmentSortTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", blockNum: " << blockNum_;
    info << ", blockTilingSize: " << blockTilingSize_;
    info << ", tailBlockTilingSize: " << tailBlockTilingSize_;
    info << ", innerDim: " << innerDim_;
    info << ", outputOuterDim: " << outputOuterDim_;
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", padM: " << padM_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdSegmentSortTiling, 17);

} // namespace optiling
