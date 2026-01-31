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
 * \file segment_sum_simt_tiling.cpp
 * \brief segment_sum_simt_tiling
 */
 
#include "segment_sum_simt_tiling.h"

namespace optiling {

static constexpr uint64_t SIMT_DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint64_t TEMPLATE_SIMT = 1000;

bool SegmentSumSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus SegmentSumSimtTiling::DoOpTiling()
{
    ubSize_ -= SIMT_DCACHE_SIZE;
    uint64_t outputSize = segmentNum_ * innerDim_;
    initNumPerCore_ = outputSize / totalCoreNum_;
    initNumTailCore_ = outputSize - (totalCoreNum_ - 1) * initNumPerCore_;
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SegmentSumSimtTiling::SetTilingData()
{
    tilingData_ = context_->GetTilingData<SegmentSumSimtTilingData>();
    tilingData_->outerDim = outerDim_;
    tilingData_->innerDim = innerDim_;
    tilingData_->initNumPerCore = initNumPerCore_;
    tilingData_->initNumTailCore = initNumTailCore_;
}

uint64_t SegmentSumSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMT;
    return tilingKey;
}

ge::graphStatus SegmentSumSimtTiling::PostTiling()
{
    context_->SetBlockDim(totalCoreNum_);
    context_->SetLocalMemorySize(ubSize_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void SegmentSumSimtTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", UB Size: " << ubSize_;
    info << ", totalCoreNum: " << totalCoreNum_;
    info << ", outerDim: " << outerDim_;
    info << ", innerDim: " << innerDim_;
    info << ", initNumPerCore: " << initNumPerCore_;
    info << ", initNumTailCore: " <<initNumTailCore_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("SegmentSum", SegmentSumSimtTiling, 60);
}