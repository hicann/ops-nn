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
 * \file unsorted_segment_simd_spilt_col_tiling.cpp
 * \brief unsorted_segment_simd_spilt_col_tiling
 */

#include "unsorted_segment_simd_spilt_col_tiling.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_SIMD_SPLIT_COL = 5000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 256;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t BASE_A_SIZE = 1024;
static constexpr uint64_t RATIO_BY_SORT = 5;

bool UnsortedSegmentSimdSplitColTiling::IsCapable()
{
    if (innerDim_ * dataTypeBytes_ > totalCoreNum_ * LAST_DIM_SIMD_COND && ratio_ > RATIO_BY_SORT) {
        return IsFullLoad();
    }
    return false;
}

uint64_t UnsortedSegmentSimdSplitColTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMD_SPLIT_COL;
    return tilingKey;
}

void UnsortedSegmentSimdSplitColTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentSimdSplitColTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimdSplitColTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->normBlockData = normBlockData_;
    tilingData->tailBlockData = tailBlockData_;
    tilingData->baseS = baseS_;
    tilingData->baseA = baseA_;
}

bool UnsortedSegmentSimdSplitColTiling::IsFullLoad()
{
    normBlockData_ = Ops::Base::CeilDiv(innerDim_, totalCoreNum_);
    normBlockData_ = Ops::Base::CeilAlign(normBlockData_, ubBlockSize_ / dataTypeBytes_);
    normBlockData_ = std::max(normBlockData_, LAST_DIM_SIMD_COND / dataTypeBytes_);
    usedCoreNum_ = Ops::Base::CeilDiv(innerDim_, normBlockData_);
    tailBlockData_ = innerDim_ - (usedCoreNum_ - 1UL) * normBlockData_;

    baseA_ = std::min(BASE_A_SIZE / dataTypeBytes_, normBlockData_);
    baseS_ = 1UL;
    outUbsize_ = outputOuterDim_ * baseA_ * dataTypeBytes_;
    uint64_t needUbSize = outUbsize_ + baseS_ * baseA_ * dataTypeBytes_ * BUFFER_NUM +
                          (baseS_ * idTypeBytes_ + ubBlockSize_) * BUFFER_NUM;
    if (needUbSize < ubSize_) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentSimdSplitColTiling::DoOpTiling()
{
    baseS_ =
        (ubSize_ - outUbsize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM / (baseA_ * dataTypeBytes_ + idTypeBytes_);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSimdSplitColTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSimdSplitColTiling::DumpTilingInfo()
{
    UnsortedSegment::UnsortedSegmentSimdSplitColTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimdSplitColTilingData>();
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", normBlockData: " << tilingData->normBlockData;
    info << ", tailBlockData: " << tilingData->tailBlockData;
    info << ", baseS: " << tilingData->baseS;
    info << ", baseA: " << tilingData->baseA;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling