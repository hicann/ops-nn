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
 * \file unsorted_segment_sum_deterministic_small_innerdim_tiling.cpp
 * \brief unsorted_segment_sum_deterministic_small_innerdim_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"
#include "unsorted_segment_sum_deterministic_small_innerdim_tiling.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_DETERMINISTIC_SMALL_INNERDIM = 9000;
static constexpr uint32_t DCACHE_SIZE = 32 * 1024;
static constexpr uint32_t NUMBER_THREE = 3;
static constexpr uint32_t DOUBLE = 2;
static constexpr uint64_t INPUT_OUTER_DIM_THRESHOLD = 10000;

static const std::set<ge::DataType> deterministicSupportType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool UnsortedSegmentSumDetermSmallInnerDimTiling::IsCapable()
{
    OP_LOGI(
        context_->GetNodeName(), "[UnsortedSegmentSum] DetermSmallInnerDim GetDeterministic state: %u",
        context_->GetDeterministic());
    if (context_->GetDeterministic() != 1 || dataShapeSize_ == 0 || innerDim_ != 1 ||
        inputOuterDim_ >= INPUT_OUTER_DIM_THRESHOLD ||
        deterministicSupportType.find(dataType_) == deterministicSupportType.end()) {
        return false;
    }
    uint64_t savedUbSize = ubSize_;
    ubSize_ -= DCACHE_SIZE;
    ubSize_ -= NUMBER_THREE * ubBlockSize_;
    int64_t rowsNumInUB = FindMaxRowsInUb();
    ubSize_ = savedUbSize;
    return rowsNumInUB > 0;
}

int64_t UnsortedSegmentSumDetermSmallInnerDimTiling::FindMaxRowsInUb()
{
    uint64_t oneRowOccupyBytes = valueTypeBytes_ + idTypeBytes_ * DOUBLE + sizeof(uint32_t);

    int64_t left = 0;
    int64_t right = ubSize_ / oneRowOccupyBytes;
    int64_t maxRows = -1;

    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        int64_t occupyBytes = mid * oneRowOccupyBytes + GetSortTmpSize(mid);
        if (occupyBytes <= static_cast<int64_t>(ubSize_)) {
            maxRows = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return maxRows;
}

int64_t UnsortedSegmentSumDetermSmallInnerDimTiling::GetSortTmpSize(int64_t rowsNum)
{
    std::vector<int64_t> shapeVec = {rowsNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, idType_, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
}

void UnsortedSegmentSumDetermSmallInnerDimTiling::SetTilingData()
{
    tilingData_.set_inputOuterDim(inputOuterDim_);
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_rowsNumInUB(rowsNumInUB_);
    tilingData_.set_sortSharedBufSize(sortSharedBufSize_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
}

ge::graphStatus UnsortedSegmentSumDetermSmallInnerDimTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "Deterministic Small InnerDim Mode tiling is begin");
    ubSize_ -= DCACHE_SIZE;
    ubSize_ -= NUMBER_THREE * ubBlockSize_;

    int64_t rowsNumInUB = FindMaxRowsInUb();
    if (rowsNumInUB <= 0) {
        OP_LOGW(context_->GetNodeName(), "Cannot fit any rows in UB, current module does not support !!!");
        return ge::GRAPH_PARAM_INVALID;
    }

    rowsNumInUB_ = static_cast<uint32_t>(rowsNumInUB);
    usedCoreNum_ = 1;
    sortSharedBufSize_ = static_cast<uint32_t>(GetSortTmpSize(rowsNumInUB));

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumDetermSmallInnerDimTiling::GetTilingKey() const
{
    return TEMPLATE_DETERMINISTIC_SMALL_INNERDIM;
}

ge::graphStatus UnsortedSegmentSumDetermSmallInnerDimTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %ld failed.", ubSize_), return ge::GRAPH_FAILED);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumDetermSmallInnerDimTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData_.get_inputOuterDim();
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", rowsNumInUB: " << tilingData_.get_rowsNumInUB();
    info << ", sortSharedBufSize: " << tilingData_.get_sortSharedBufSize();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumDetermSmallInnerDimTiling, 1);

} // namespace optiling