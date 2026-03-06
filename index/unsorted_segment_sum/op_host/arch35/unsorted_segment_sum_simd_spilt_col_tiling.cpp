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
 * \file unsorted_segment_sum_simd_spilt_col_tiling.cpp
 * \brief unsorted_segment_sum_simd_spilt_col_tiling
 */

#include "unsorted_segment_sum_simd_spilt_col_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_SIMD_SPLIT_COL = 5000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 256;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t SIMD_RESERVED_SIZE = 8192;
static constexpr uint64_t BASE_A_SIZE = 1024;
static constexpr uint64_t RATIO_BY_SORT = 5;

bool UnsortedSegmentSumSimdSplitColTiling::IsCapable()
{
    if (innerDim_ * valueTypeBytes_ > totalCoreNum_ * LAST_DIM_SIMD_COND && ratio_ > RATIO_BY_SORT) {
        return IsFullLoad();
    }
    return false;
}

uint64_t UnsortedSegmentSumSimdSplitColTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMD_SPLIT_COL;
    return tilingKey;
}

void UnsortedSegmentSumSimdSplitColTiling::SetTilingData()
{
    tilingData_.set_inputOuterDim(inputOuterDim_);
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_normBlockData(normBlockData_);
    tilingData_.set_tailBlockData(tailBlockData_);
    tilingData_.set_baseS(baseS_);
    tilingData_.set_baseA(baseA_);
}

bool UnsortedSegmentSumSimdSplitColTiling::IsFullLoad()
{
    normBlockData_ = Ops::Base::CeilDiv(innerDim_, totalCoreNum_);
    normBlockData_ = Ops::Base::CeilAlign(normBlockData_, ubBlockSize_ / valueTypeBytes_);
    normBlockData_ = std::max(normBlockData_, LAST_DIM_SIMD_COND / valueTypeBytes_);
    usedCoreNum_ = Ops::Base::CeilDiv(innerDim_, normBlockData_);
    tailBlockData_ = innerDim_ - (usedCoreNum_ - 1UL) * normBlockData_;

    ubSize_ -= SIMD_RESERVED_SIZE;
    baseA_ = std::min(BASE_A_SIZE / valueTypeBytes_, normBlockData_);
    baseS_ = 1UL;
    outUbsize_ = outputOuterDim_ * baseA_ * valueTypeBytes_;
    uint64_t needUbSize = outUbsize_ + baseS_ * baseA_ * valueTypeBytes_ * BUFFER_NUM +
                          (baseS_ * idTypeBytes_ + ubBlockSize_) * BUFFER_NUM;
    if (needUbSize < ubSize_) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentSumSimdSplitColTiling::DoOpTiling()
{
    baseS_ =
        (ubSize_ - outUbsize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM / (baseA_ * valueTypeBytes_ + idTypeBytes_);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumSimdSplitColTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumSimdSplitColTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData_.get_inputOuterDim();
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", normBlockData: " << tilingData_.get_normBlockData();
    info << ", tailBlockData: " << tilingData_.get_tailBlockData();
    info << ", baseS: " << tilingData_.get_baseS();
    info << ", baseA: " << tilingData_.get_baseA();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumSimdSplitColTiling, 20);

} // namespace optiling