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
 * \file unsorted_segment_sort_simt_tiling.cpp
 * \brief unsorted_segment_sort_simt_tiling
 */
#include "unsorted_segment_sort_simt_tiling.h"

namespace optiling {
static constexpr uint32_t IN_OUT_RATE_THRESHOLD = 5;
static constexpr uint32_t INNER_DIM_THRESHOLD = 128;
static constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint32_t MAX_INDEX_NUM = 1024;
static constexpr int64_t DOUBLE = 2;
static constexpr uint64_t TEMPLATE_SORT_SIMT = 4100;
static constexpr uint32_t ALIGN_SIZE = 128;

bool UnsortedSegmentSortSimtTiling::IsCapable() 
{
    if (inputOuterDim_ / outputOuterDim_ >= IN_OUT_RATE_THRESHOLD && innerDim_ * dataTypeBytes_ < INNER_DIM_THRESHOLD) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentSortSimtTiling::DoOpTiling()
{
    ge::graphStatus ret = CalcTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSortSimtTiling::CalcTiling()
{
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(inputOuterDim_) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    ubSize_ = ubSize_ - DCACHE_SIZE;
    while (end - start > 1) {
        mid = (end + start) / DOUBLE;
        int64_t totalIndexSize = Ops::Base::CeilAlign(mid * idTypeBytes_, ubBlockSize_) * DOUBLE +
                                 Ops::Base::CeilAlign(mid * idTypeBytes_, ubBlockSize_) + ubBlockSize_ * DOUBLE +
                                 Ops::Base::CeilAlign(mid * sizeof(uint32_t), ubBlockSize_);
        sortTmpSize = GetSortTmpSize(mid, false);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, static_cast<int64_t>(ubBlockSize_));
        int64_t tmpTotalSize =
            totalIndexSize + sortTmpSize + Ops::Base::CeilAlign(mid * innerDim_ * dataTypeBytes_, ubBlockSize_) * DOUBLE;
        if (tmpTotalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }
    int64_t rowUb = start;
    int64_t totalLoop = Ops::Base::CeilDiv(static_cast<int64_t>(inputOuterDim_), rowUb);
    int64_t eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<int64_t>(totalCoreNum_));
    usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
    int64_t tailCoreLoop = totalLoop - eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1);
    int64_t tailIndexNum = static_cast<int64_t>(inputOuterDim_) -
                           rowUb * eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1) - rowUb * (tailCoreLoop - 1);
    while (usedCoreNum_ < (totalCoreNum_ / static_cast<uint64_t>(DOUBLE)) && rowUb > 1) {
        rowUb = rowUb / DOUBLE;
        totalLoop = Ops::Base::CeilDiv(static_cast<int64_t>(inputOuterDim_), rowUb);
        eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<int64_t>(totalCoreNum_));
        usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
        tailCoreLoop = totalLoop - eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1);
        tailIndexNum = static_cast<int64_t>(inputOuterDim_) -
                       rowUb * eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1) - rowUb * (tailCoreLoop - 1);
    }

    UnsortedSegment::UnsortedSegmentSortSimtTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSortSimtTilingData>();

    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->maxIndexNum = rowUb;
    tilingData->oneCoreUbLoopTimes = eachCoreLoop;
    tilingData->tailCoreUbLoopTimes = tailCoreLoop;
    tilingData->maxThread = maxThread_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->sortTmpSize = sortTmpSize;
    tilingData->tailIndexNum = tailIndexNum;
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSortSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SORT_SIMT;
    return tilingKey;
}

ge::graphStatus UnsortedSegmentSortSimtTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSortSimtTiling::DumpTilingInfo()
{   
    UnsortedSegment::UnsortedSegmentSortSimtTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSortSimtTilingData>();
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", maxIndexNum: " << tilingData->maxIndexNum;
    info << ", oneCoreUbLoopTimes: " << tilingData->oneCoreUbLoopTimes;
    info << ", tailCoreUbLoopTimes: " << tilingData->tailCoreUbLoopTimes;
    info << ", maxThread: " << tilingData->maxThread;
    info << ", usedCoreNum: " << tilingData->usedCoreNum;
    info << ", sortTmpSize: " << tilingData->sortTmpSize;
    info << ", tailIndexNum: " << tilingData->tailIndexNum;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling