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
 * \file unsorted_segment_sum_output_fullload_tiling.cpp
 * \brief unsorted_segment_sum_output_fullload_tiling
 */
#include "unsorted_segment_sum_output_fullload_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace UnsortedSegmentSum;
static constexpr uint32_t ROW_NUM = 16;
static constexpr uint32_t MAX_LAST_DIM_RANG = 1024;
static constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint64_t BUFFER_ADD_NUM = 2;
static constexpr uint64_t UB_MIN_FACTOR = 2048;
static constexpr uint32_t SIMD_RESERVED_SIZE = 8192;

bool UnsortedSegmentSumOutFlTiling::IsCapable()
{
    if ((isSupportDtype() && isSupportSize()) || dataShapeSize_ == 0UL) {
        return true;
    }
    return false;
}

bool UnsortedSegmentSumOutFlTiling::isSupportDtype()
{
    if (dataType_ != ge::DT_INT64 && dataType_ != ge::DT_UINT64 && dataType_ != ge::DT_UINT32) {
        return true;
    }
    return false;
}

bool UnsortedSegmentSumOutFlTiling::isSupportSize()
{
    uint32_t lastDimRange = MAX_LAST_DIM_RANG / ROW_NUM;
    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t xBufferSize = ROW_NUM * innerDim_ * valueTypeBytes_ + ubBlockSize_;
    uint64_t indexBufferSize = ROW_NUM * idTypeBytes_ + ubBlockSize_;
    uint64_t tmpBufferSize = Ops::Base::CeilAlign(outSize * valueTypeBytes_, ubBlockSize_) * ROW_NUM;
    uint64_t needUb = xBufferSize * BUFFER_ADD_NUM + indexBufferSize * BUFFER_ADD_NUM + tmpBufferSize;
    ubSize_ = ubSize_ - DCACHE_SIZE - SIMD_RESERVED_SIZE;
    if (needUb < ubSize_ && innerDimAlign_ <= lastDimRange) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentSumOutFlTiling::DoOpTiling()
{
    UbAddBranch();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumOutFlTiling::UbAddBranch()
{
    if (dataShapeSize_ == 0UL) {
        uint64_t outSize = outputOuterDim_ * innerDim_;
        uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
        normBlockData = std::max(normBlockData, UB_MIN_FACTOR / valueTypeBytes_);
        usedCoreNum_ = Ops::Base::CeilDiv(outSize, normBlockData);
        return ge::GRAPH_SUCCESS;
    }
    maxIndexNum_ = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
    if (1UL == maxIndexNum_) {
        usedCoreNum_ = std::min(inputOuterDim_, totalCoreNum_);
    } else {
        usedCoreNum_ = std::min(Ops::Base::CeilDiv(inputOuterDim_, maxIndexNum_), totalCoreNum_);
    }
    uint64_t oneRowinnerDimSize = innerDim_ * valueTypeBytes_;
    uint64_t outTailSize = Ops::Base::CeilAlign(outputOuterDim_ * innerDim_ * valueTypeBytes_, ubBlockSize_) * ROW_NUM;
    uint64_t remainUbSize = (ubSize_ - outTailSize) / BUFFER_ADD_NUM;
    // one ubBlockSize_ is left for input align
    rowNumUb_ = (remainUbSize - ubBlockSize_ - ubBlockSize_) / (oneRowinnerDimSize + idTypeBytes_);
    oneCoreUbLoopTimes_ = (maxIndexNum_ + rowNumUb_ - 1UL) / rowNumUb_;
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumOutFlTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_OUT_FL, USS_CAST_NONE);
    return tilingKey;
}

void UnsortedSegmentSumOutFlTiling::SetTilingData()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumOutFlTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->maxIndexNum = maxIndexNum_;
    tilingData->oneCoreUbLoopTimes = oneCoreUbLoopTimes_;
    tilingData->rowNumUb = rowNumUb_;
}

ge::graphStatus UnsortedSegmentSumOutFlTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumOutFlTiling::DumpTilingInfo()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumOutFlTilingData>();
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", maxIndexNum: " << tilingData->maxIndexNum;
    info << ", oneCoreUbLoopTimes: " << tilingData->oneCoreUbLoopTimes;
    info << ", rowNumUb: " << tilingData->rowNumUb;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentSum", UnsortedSegmentSumOutFlTiling, 10);
} // namespace optiling
