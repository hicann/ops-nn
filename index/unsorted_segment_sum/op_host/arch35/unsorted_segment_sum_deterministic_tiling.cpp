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
 * \file unsorted_segment_sum_deterministic_tiling.cpp
 * \brief unsorted_segment_sum_deterministic_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"
#include "unsorted_segment_sum_deterministic_tiling.h"

namespace optiling {
using namespace UnsortedSegmentSum;
static constexpr uint32_t DCACHE_SIZE = 32 * 1024;
static constexpr uint32_t FLOAT_BYTES = 4;
static constexpr uint32_t DOUBLE = 2;
static constexpr uint32_t NUMBER_THREE = 3;
static constexpr uint32_t NUMBER_KB = 1024;
static constexpr uint32_t BIG_INNERDIM_THRESHOLD = 30;
static constexpr uint32_t SMALL_INNERDIM_THRESHOLD = 10000;

static const std::set<ge::DataType> determinsiticSupportType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool UnsortedSegmentSumDetermTiling::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "[UnsortedSegmentSum] GetDeterministic state: %u", context_->GetDeterministic());
    if (context_->GetDeterministic() == 1 && dataShapeSize_ != 0 &&
        determinsiticSupportType.find(dataType_) != determinsiticSupportType.end() &&
        !(innerDim_ > BIG_INNERDIM_THRESHOLD) && !(innerDim_ == 1 && inputOuterDim_ < SMALL_INNERDIM_THRESHOLD)) {
        return true;
    }
    return false;
}

uint64_t UnsortedSegmentSumDetermTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_DETERM, USS_CAST_NONE);
    return tilingKey;
}

void UnsortedSegmentSumDetermTiling::SetTilingData()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumDetermTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->tmpBufferSize = sortSharedBufSize_;
    tilingData->rowsNumInUB = rowsNumInUB_;
    tilingData->normalCoreProcessNum = normalCoreProcessNum_;
    tilingData->tailCoreProcessNum = tailCoreProcessNum_;
    tilingData->usedCoreNum = usedCoreNum_;
}

int64_t UnsortedSegmentSumDetermTiling::FindMaxRowsInUb()
{
    uint32_t alignNum = ubBlockSize_ / sizeof(float);
    row32BAlign_ = (innerDim_ + alignNum - 1) / alignNum * alignNum;
    ubSize_ -= NUMBER_THREE * ubBlockSize_;
    uint64_t oneRowOccupyBytes = valueTypeBytes_ * row32BAlign_ // x输入
                                 + idTypeBytes_ * DOUBLE        // segmentId输入、sortedId
                                 + sizeof(uint32_t);            // sortedIdx
    int64_t left = 0;
    int64_t right = ubSize_ / oneRowOccupyBytes;
    int64_t maxRows = -1;

    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
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

int64_t UnsortedSegmentSumDetermTiling::GetSortTmpSize(int64_t rowsNum)
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

ge::graphStatus UnsortedSegmentSumDetermTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "Deterministic Mode tiling is begin");
    ubSize_ -= DCACHE_SIZE;
    int64_t rowsNumInUB = FindMaxRowsInUb();
    if (rowsNumInUB <= 0) {
        OP_LOGW(context_->GetNodeName(), "InnerDim is too large, current module does not support !!!");
        return ge::GRAPH_PARAM_INVALID;
    }

    if (rowsNumInUB >= static_cast<int64_t>(inputOuterDim_)) {
        usedCoreNum_ = 1;
        normalCoreProcessNum_ = inputOuterDim_;
        tailCoreProcessNum_ = inputOuterDim_;
    } else {
        normalCoreProcessNum_ = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(inputOuterDim_, normalCoreProcessNum_);
        tailCoreProcessNum_ = inputOuterDim_ - (usedCoreNum_ - 1) * normalCoreProcessNum_;
    }
    uint64_t perCoreSortedIdBytes = (normalCoreProcessNum_ * idTypeBytes_ + 31) / 32 * 32;
    uint64_t perCoreSortedIndexBytes = (normalCoreProcessNum_ * sizeof(uint32_t) + 31) / 32 * 32;
    usrWorkspaceSize_ = outputOuterDim_ * innerDim_ * sizeof(int32_t) * DOUBLE + outputOuterDim_ * sizeof(int64_t) +
                        DOUBLE * NUMBER_KB + NUMBER_KB + usedCoreNum_ * perCoreSortedIdBytes + NUMBER_KB +
                        usedCoreNum_ * perCoreSortedIndexBytes;
    sortSharedBufSize_ = GetSortTmpSize(rowsNumInUB);
    rowsNumInUB_ = rowsNumInUB;
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumDetermTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(static_cast<uint32_t>(totalCoreNum_));
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumDetermTiling::DumpTilingInfo()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumDetermTilingData>();
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", tmpBufferSize: " << tilingData->tmpBufferSize;
    info << ", rowsNumInUB: " << tilingData->rowsNumInUB;
    info << ", normalCoreProcessNum: " << tilingData->normalCoreProcessNum;
    info << ", tailCoreProcessNum: " << tilingData->tailCoreProcessNum;
    info << ", usedCoreNum: " << tilingData->usedCoreNum;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentSum", UnsortedSegmentSumDetermTiling, 0);

} // namespace optiling
