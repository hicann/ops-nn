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
 * \file unsorted_segment_simt_tiling.cpp
 * \brief unsorted_segment_simt_tiling
 */

#include "unsorted_segment_simt_tiling.h"

namespace optiling {
static constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint64_t TEMPLATE_SIMT = 1000;
static constexpr uint64_t UB_MIN_FACTOR = 2048;

bool UnsortedSegmentSimtTiling::IsCapable()
{
    return true;
}

uint64_t UnsortedSegmentSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMT;
    return tilingKey;
}

void UnsortedSegmentSimtTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentSimtTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimtTilingData>();

    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->maxThread = static_cast<uint64_t>(maxThread_);
}

ge::graphStatus UnsortedSegmentSimtTiling::DoOpTiling()
{
    usedCoreNum_ = Ops::Base::CeilDiv(dataShapeSize_, static_cast<uint64_t>(maxThread_));
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
    normBlockData = std::max(normBlockData, UB_MIN_FACTOR / dataTypeBytes_);
    uint64_t initUsedCore = Ops::Base::CeilDiv(outSize, normBlockData);
    usedCoreNum_ = std::max(usedCoreNum_, initUsedCore);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus UnsortedSegmentSimtTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    ubSize_ = ubSize_ - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_TILING_CHECK(
        (res != ge::GRAPH_SUCCESS),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize_),
        return ge::GRAPH_FAILED);
        
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSimtTiling::DumpTilingInfo()
{
    UnsortedSegment::UnsortedSegmentSimtTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimtTilingData>();

    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", maxThread: " << tilingData->maxThread;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling