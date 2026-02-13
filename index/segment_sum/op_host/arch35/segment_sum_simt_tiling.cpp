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
static constexpr uint32_t DOUBLE = 2;
static constexpr uint32_t ROWS_IN_WORKSPACE = 128;
static constexpr uint32_t RESERVED_WS_SIZE = 16 * 1024 * 1024;
static const std::set<ge::DataType> deterministicType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool SegmentSumSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus SegmentSumSimtTiling::DoOpTiling()
{
    ubSize_ -= SIMT_DCACHE_SIZE;
    if (context_->GetDeterministic() && deterministicType.find(dataType_) != deterministicType.end()) {
        isDeterministic_ = 1;
    }
    uint64_t outputSize = segmentNum_ * innerDim_;
    initNumPerCore_ = outputSize / totalCoreNum_;
    initNumTailCore_ = outputSize - (totalCoreNum_ - 1) * initNumPerCore_;
    
    segIdsPerCore_ = outerDim_ / totalCoreNum_;
    segIdsTailCore_ = outerDim_ - segIdsPerCore_ * (totalCoreNum_-1);
    maxSegIdsInUb = (ubSize_ / DOUBLE - ubBlockSize_) / idTypeBytes_;
    segIdsPerLoop_ = maxSegIdsInUb > segIdsPerCore_ ? segIdsPerCore_ : maxSegIdsInUb;
    segIdsPerLoopTailCore_ = maxSegIdsInUb > segIdsTailCore_ ? segIdsTailCore_ : maxSegIdsInUb;
    loopTimes_ = Ops::Base::CeilDiv(segIdsPerCore_, static_cast<uint64_t>(segIdsPerLoop_));
    loopTimesTailCore_ = Ops::Base::CeilDiv(segIdsTailCore_, static_cast<uint64_t>(segIdsPerLoopTailCore_));
    segIdsTailLoop_ = segIdsPerCore_ - (loopTimes_ - 1) * segIdsPerLoop_;
    segIdsTailLoopTailCore_ = segIdsTailCore_ - (loopTimesTailCore_ - 1) * segIdsPerLoopTailCore_;

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
    tilingData_->isDeterministic = isDeterministic_;
    tilingData_->maxSegIdsInUb = maxSegIdsInUb;
    tilingData_->loopTimes = loopTimes_;
    tilingData_->loopTimesTailCore = loopTimesTailCore_;
    tilingData_->segIdsPerLoop = segIdsPerLoop_;
    tilingData_->segIdsPerLoopTailCore = segIdsPerLoopTailCore_;
    tilingData_->segIdsTailLoop = segIdsTailLoop_;
    tilingData_->segIdsTailLoopTailCore = segIdsTailLoopTailCore_;
}

uint64_t SegmentSumSimtTiling::GetTilingKey() const
{
    return TEMPLATE_SIMT;
}

ge::graphStatus SegmentSumSimtTiling::GetWorkspaceSize()
{
    auto currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = RESERVED_WS_SIZE;
    if (isDeterministic_ == 1) {
        uint64_t ws_size = ROWS_IN_WORKSPACE * innerDim_ * valueTypeBytes_ + 
                           ROWS_IN_WORKSPACE * idTypeBytes_;
        currentWorkspace[0] += ws_size ;    
    }
    return ge::GRAPH_SUCCESS;
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
    info << ", initNumTailCore: " << initNumTailCore_;
    info << ", isDeterministic: " << isDeterministic_;
    info << ", segIdsPerCore: " << segIdsPerCore_;
    info << ", segIdsTailCore: " << segIdsTailCore_;
    info << ", segIdsPerLoop: " << segIdsPerLoop_;
    info << ", segIdsPerLoopTailCore: " << segIdsPerLoopTailCore_;
    info << ", segIdsTailLoop: " << segIdsTailLoop_;
    info << ", segIdsTailLoopTailCore: " << segIdsTailLoopTailCore_;
    info << ", loopTimes: " << loopTimes_;
    info << ", loopTimesTailCore: " << loopTimesTailCore_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("SegmentSum", SegmentSumSimtTiling, 60);
}