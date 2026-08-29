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
using namespace UnsortedSegmentSum;
static constexpr uint64_t LAST_DIM_SIMD_COND = 256;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t SIMD_RESERVED_SIZE = 8192;
static constexpr uint64_t BASE_A_SIZE = 1024;

bool UnsortedSegmentSumSimdSplitColTiling::IsCapable()
{
    if (inputOuterDim_ < totalCoreNum_ && innerDim_ * valueTypeBytes_ > totalCoreNum_ * LAST_DIM_SIMD_COND) {
        return IsFullLoad();
    }
    return false;
}

uint64_t UnsortedSegmentSumSimdSplitColTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(USS_TEMPLATE_SIMD_SPLIT_COL, USS_CAST_NONE);
}

void UnsortedSegmentSumSimdSplitColTiling::SetTilingData()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumSimdSplitColTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->normBlockData = normBlockData_;
    tilingData->tailBlockData = tailBlockData_;
    tilingData->baseS = baseS_;
    tilingData->baseA = baseA_;
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
    baseS_ = (ubSize_ - outUbsize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM /
             (baseA_ * valueTypeBytes_ + idTypeBytes_);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumSimdSplitColTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumSimdSplitColTiling::DumpTilingInfo()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumSimdSplitColTilingData>();
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

REGISTER_TILING_TEMPLATE("UnsortedSegmentSum", UnsortedSegmentSumSimdSplitColTiling, 20);

} // namespace optiling
