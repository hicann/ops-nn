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
 * \file unsorted_segment_simd_non_sort_tiling.cpp
 * \brief unsorted_segment_simd_non_sort_tiling
 */

#include "unsorted_segment_simd_non_sort_tiling.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_SIMD_NON_SORT = 6000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 128;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t RATIO_BY_SORT = 5;
static constexpr uint64_t UB_MIN_FACTOR = 2048;

static const std::set<ge::DataType> setAtomicNotSupport = {ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};

bool UnsortedSegmentSimdNonSortTiling::IsCapable()
{
    if (ratio_ < RATIO_BY_SORT && innerDim_ * dataTypeBytes_ >= LAST_DIM_SIMD_COND &&
        setAtomicNotSupport.find(dataType_) == setAtomicNotSupport.end()) {
        return true;
    }
    return false;
}

uint64_t UnsortedSegmentSimdNonSortTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMD_NON_SORT;
    return tilingKey;
}

void UnsortedSegmentSimdNonSortTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentSimdNonSortTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimdNonSortTilingData>();

    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->sTileNum = sTileNum_;
    tilingData->aTileNum = aTileNum_;
    tilingData->normBlockS = normBlockS_;
    tilingData->tailBlockS = tailBlockS_;
    tilingData->normBlockA = normBlockA_;
    tilingData->tailBlockA = tailBlockA_;
    tilingData->baseS = baseS_;
    tilingData->baseA = baseA_;
    tilingData->usedCoreNum = usedCoreNum_;
}

ge::graphStatus UnsortedSegmentSimdNonSortTiling::DoOpTiling()
{
    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
    normBlockData = std::max(normBlockData, UB_MIN_FACTOR / dataTypeBytes_);
    initUsedCore_ = Ops::Base::CeilDiv(outSize, normBlockData);

    // block split, ensure that ub prioritizes a, block split s * (a/maxBaseA)
    uint64_t minBaseS = 1;
    uint64_t maxBaseA =
        (ubSize_ - BUFFER_NUM * (ubBlockSize_ + minBaseS * idTypeBytes_)) / BUFFER_NUM / (minBaseS * dataTypeBytes_);
    maxBaseA = Ops::Base::FloorAlign(maxBaseA, ubBlockSize_ / dataTypeBytes_);

    uint64_t colNumAlign = Ops::Base::CeilDiv(innerDim_, maxBaseA);
    usedCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>(inputOuterDim_ * colNumAlign));
    usedCoreNum_ = usedCoreNum_ == 0UL ? 1UL : usedCoreNum_;
    std::tie(sTileNum_, aTileNum_) = AutoTiling(usedCoreNum_, colNumAlign, maxBaseA * dataTypeBytes_);

    normBlockS_ = Ops::Base::CeilDiv(inputOuterDim_, sTileNum_);
    sTileNum_ = Ops::Base::CeilDiv(inputOuterDim_, normBlockS_);

    normBlockA_ = Ops::Base::CeilDiv(innerDim_, aTileNum_);
    normBlockA_ = Ops::Base::CeilAlign(normBlockA_, ubBlockSize_ / dataTypeBytes_);
    aTileNum_ = Ops::Base::CeilDiv(innerDim_, normBlockA_);

    usedCoreNum_ = sTileNum_ * aTileNum_;

    tailBlockS_ = inputOuterDim_ - (sTileNum_ - 1UL) * normBlockS_;
    tailBlockA_ = innerDim_ - (aTileNum_ - 1UL) * normBlockA_;

    // ub split
    baseA_ = std::min(normBlockA_, maxBaseA);
    baseS_ = (ubSize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM / (baseA_ * dataTypeBytes_ + idTypeBytes_);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSimdNonSortTiling::PostTiling()
{
    context_->SetBlockDim(std::max(usedCoreNum_, initUsedCore_));
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSimdNonSortTiling::DumpTilingInfo()
{
    UnsortedSegment::UnsortedSegmentSimdNonSortTilingData *tilingData = 
        context_->GetTilingData<UnsortedSegment::UnsortedSegmentSimdNonSortTilingData>();

    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", initUsedCore: " << initUsedCore_;
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", sTileNum: " << tilingData->sTileNum;
    info << ", aTileNum: " << tilingData->aTileNum;
    info << ", normBlockS: " << tilingData->normBlockS;
    info << ", tailBlockS: " << tilingData->tailBlockS;
    info << ", normBlockA: " << tilingData->normBlockA;
    info << ", tailBlockA: " << tilingData->tailBlockA;
    info << ", baseS: " << tilingData->baseS;
    info << ", baseA: " << tilingData->baseA;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling