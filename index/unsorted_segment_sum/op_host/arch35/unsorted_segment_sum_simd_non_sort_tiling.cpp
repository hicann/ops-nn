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
 * \file unsorted_segment_sum_simd_non_sort_tiling.cpp
 * \brief unsorted_segment_sum_simd_non_sort_tiling
 */

#include "unsorted_segment_sum_simd_non_sort_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"



#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_SIMD_NON_SORT = 6000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 128;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t RATIO_BY_SORT = 5;
static constexpr uint64_t SIMD_RESERVED_SIZE = 8192;
static constexpr uint64_t UB_MIN_FACTOR = 2048;

static const std::set<ge::DataType> setAtomicNotSupport = {ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};

bool UnsortedSegmentSumSimdNonSortTiling::IsCapable()
{
    if (ratio_ < RATIO_BY_SORT && innerDim_ * valueTypeBytes_ >= LAST_DIM_SIMD_COND &&
        setAtomicNotSupport.find(dataType_) == setAtomicNotSupport.end()) {
        return true;
    }
    return false;
}

uint64_t UnsortedSegmentSumSimdNonSortTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMD_NON_SORT;
    return tilingKey;
}

void UnsortedSegmentSumSimdNonSortTiling::SetTilingData()
{
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_sTileNum(sTileNum_);
    tilingData_.set_aTileNum(aTileNum_);
    tilingData_.set_normBlockS(normBlockS_);
    tilingData_.set_tailBlockS(tailBlockS_);
    tilingData_.set_normBlockA(normBlockA_);
    tilingData_.set_tailBlockA(tailBlockA_);
    tilingData_.set_baseS(baseS_);
    tilingData_.set_baseA(baseA_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
}

ge::graphStatus UnsortedSegmentSumSimdNonSortTiling::DoOpTiling()
{
    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
    normBlockData = std::max(normBlockData, UB_MIN_FACTOR / valueTypeBytes_);
    initUsedCore_ = Ops::Base::CeilDiv(outSize, normBlockData);

    // block split, ensure that ub prioritizes a, block split s * (a/maxBaseA)
    uint64_t minBaseS = 1;
    ubSize_ -= SIMD_RESERVED_SIZE;
    uint64_t maxBaseA =
        (ubSize_ - BUFFER_NUM * (ubBlockSize_ + minBaseS * idTypeBytes_)) / BUFFER_NUM / (minBaseS * valueTypeBytes_);
    maxBaseA = Ops::Base::FloorAlign(maxBaseA, ubBlockSize_ / valueTypeBytes_);

    uint64_t colNumAlign = Ops::Base::CeilDiv(innerDim_, maxBaseA);
    usedCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>(inputOuterDim_ * colNumAlign));
    usedCoreNum_ = usedCoreNum_ == 0UL ? 1UL : usedCoreNum_;
    std::tie(sTileNum_, aTileNum_) = AutoTiling(usedCoreNum_, colNumAlign, maxBaseA * valueTypeBytes_);

    normBlockS_ = Ops::Base::CeilDiv(inputOuterDim_, sTileNum_);
    sTileNum_ = Ops::Base::CeilDiv(inputOuterDim_, normBlockS_);

    normBlockA_ = Ops::Base::CeilDiv(innerDim_, aTileNum_);
    normBlockA_ = Ops::Base::CeilAlign(normBlockA_, ubBlockSize_ / valueTypeBytes_);
    aTileNum_ = Ops::Base::CeilDiv(innerDim_, normBlockA_);

    usedCoreNum_ = sTileNum_ * aTileNum_;

    tailBlockS_ = inputOuterDim_ - (sTileNum_ - 1UL) * normBlockS_;
    tailBlockA_ = innerDim_ - (aTileNum_ - 1UL) * normBlockA_;

    // ub split
    baseA_ = std::min(normBlockA_, maxBaseA);
    baseS_ = (ubSize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM / (baseA_ * valueTypeBytes_ + idTypeBytes_);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumSimdNonSortTiling::PostTiling()
{
    context_->SetBlockDim(std::max(usedCoreNum_, initUsedCore_));
    context_->SetScheduleMode(1);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumSimdNonSortTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", initUsedCore: " << initUsedCore_;
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", sTileNum: " << tilingData_.get_sTileNum();
    info << ", aTileNum: " << tilingData_.get_aTileNum();
    info << ", normBlockS: " << tilingData_.get_normBlockS();
    info << ", tailBlockS: " << tilingData_.get_tailBlockS();
    info << ", normBlockA: " << tilingData_.get_normBlockA();
    info << ", tailBlockA: " << tilingData_.get_tailBlockA();
    info << ", baseS: " << tilingData_.get_baseS();
    info << ", baseA: " << tilingData_.get_baseA();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumSimdNonSortTiling, 30);

} // namespace optiling