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
 * \file unsorted_segment_sum_deterministic_big_innerdim_tiling.cpp
 * \brief unsorted_segment_sum_deterministic_big_innerdim_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"
#include "unsorted_segment_sum_deterministic_big_innerdim_tiling.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_DETERMINISTIC_BIG_INNERDIM = 8000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 256;
static constexpr uint64_t BASE_A_SIZE = 1024;
static constexpr uint32_t DCACHE_SIZE = 32 * 1024;
static constexpr uint64_t DEFAULT_HANDLE_ROWS = 20UL;
static constexpr uint32_t NUMBER_THREE = 3;
static constexpr uint32_t DOUBLE = 2;
static constexpr uint32_t HALF_UB_ALIGN = 16;;

static const std::set<ge::DataType> determinsiticSupportType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool UnsortedSegmentSumDeterministicBigInnerDimTiling::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "[UnsortedSegmentSum] GetDeterministic state: %u", context_->GetDeterministic());
    if (context_->GetDeterministic() == 1 && dataShapeSize_ != 0 &&
        determinsiticSupportType.find(dataType_) != determinsiticSupportType.end()) {
        return true;
    }
    return false;
}

int64_t UnsortedSegmentSumDeterministicBigInnerDimTiling::FindMaxColsInUb()
{
    ubSize_ -= NUMBER_THREE * ubBlockSize_;
    uint64_t oneColOccupyBytes = sizeof(float) * baseS_        // x
                                + idTypeBytes_ * baseS_* DOUBLE  // segmentId、sortedId
                                + sizeof(uint32_t) * baseS_      // sortedIdx
                                + sizeof(float) * DOUBLE;      // y double buffer                     
    int64_t left = 0;
    int64_t right = ubSize_ / oneColOccupyBytes;
    int64_t maxCols = -1;
    int64_t sortTmpSize = GetSortTmpSize(baseS_);
    while (left <= right) {
        int64_t mid = left + (right-left) / DOUBLE;        
        int64_t occupyBytes = mid * oneColOccupyBytes + sortTmpSize;
        if (occupyBytes <= static_cast<int64_t>(ubSize_)) {
            maxCols = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return maxCols;
}

void UnsortedSegmentSumDeterministicBigInnerDimTiling::SetTilingData()
{
    tilingData_.set_inputOuterDim(inputOuterDim_);
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_normalCoreProcessCols(normalCoreProcessCols_);
    tilingData_.set_tailCoreProcessCols(tailCoreProcessCols_);
    tilingData_.set_baseS(baseS_);
    tilingData_.set_baseA(baseA_);
    tilingData_.set_sortSharedBufSize(sortSharedBufSize_);
}

int64_t UnsortedSegmentSumDeterministicBigInnerDimTiling::GetSortTmpSize(int64_t rowsNum)
{
    std::vector<int64_t> shapeVec = {rowsNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, idType_, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
}

ge::graphStatus UnsortedSegmentSumDeterministicBigInnerDimTiling::DoOpTiling()
{
    normalCoreProcessCols_ = Ops::Base::CeilDiv(innerDim_, totalCoreNum_);
    normalCoreProcessCols_ = Ops::Base::CeilAlign(normalCoreProcessCols_, ubBlockSize_ / sizeof(float));
    normalCoreProcessCols_ = std::max(normalCoreProcessCols_, LAST_DIM_SIMD_COND / sizeof(float));
    usedCoreNum_ = Ops::Base::CeilDiv(innerDim_, normalCoreProcessCols_);
    tailCoreProcessCols_ = innerDim_ - (usedCoreNum_ - 1UL) * normalCoreProcessCols_;

    baseS_ = std::min(DEFAULT_HANDLE_ROWS, inputOuterDim_);    
    int64_t colsNumInUB = FindMaxColsInUb(); 
    if (colsNumInUB <= 0) {
        OP_LOGE(
            context_->GetNodeName(), "One column data size is too large, current module does not support !!!");
        return ge::GRAPH_PARAM_INVALID;
    }
    baseA_ = colsNumInUB / HALF_UB_ALIGN * HALF_UB_ALIGN;
    sortSharedBufSize_ = GetSortTmpSize(static_cast<int64_t>(baseS_));
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumDeterministicBigInnerDimTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_DETERMINISTIC_BIG_INNERDIM;
    return tilingKey;
}

ge::graphStatus UnsortedSegmentSumDeterministicBigInnerDimTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumDeterministicBigInnerDimTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData_.get_inputOuterDim();
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", normalCoreProcessCols: " << tilingData_.get_normalCoreProcessCols();
    info << ", tailCoreProcessCols: " << tilingData_.get_tailCoreProcessCols();
    info << ", baseS: " << tilingData_.get_baseS();
    info << ", baseA: " << tilingData_.get_baseA();
    info << ", sortSharedBufSize: " << tilingData_.get_sortSharedBufSize();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumDeterministicBigInnerDimTiling, 5);
} // namespace optiling