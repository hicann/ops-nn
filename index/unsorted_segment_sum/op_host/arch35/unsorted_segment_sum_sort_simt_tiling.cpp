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
 * \file unsorted_segment_sum_sort_simt_tiling.cpp
 * \brief unsorted_segment_sum_sort_simt_tiling
 */

#include <vector>
#include "unsorted_segment_sum_sort_simt_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"


namespace optiling {
static constexpr uint32_t IN_OUT_RATE_THRESHOLD = 5;
static constexpr uint32_t INNER_DIM_THRESHOLD = 128;
static constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint32_t MAX_INDEX_NUM = 1024;
static constexpr int64_t DOUBLE = 2;
static constexpr uint64_t TEMPLATE_SORT_SIMT = 4100;
static constexpr uint32_t ALIGN_SIZE = 128;
static constexpr uint64_t SIMD_RESERVED_SIZE = 8192;

bool UnsortedSegmentSumSortSimtTiling::IsCapable() 
{
    if (inputOuterDim_ / outputOuterDim_ >= IN_OUT_RATE_THRESHOLD && innerDim_ * valueTypeBytes_ < INNER_DIM_THRESHOLD) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentSumSortSimtTiling::DoOpTiling()
{
    ge::graphStatus ret = CalcTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t UnsortedSegmentSumSortSimtTiling::GetSortBufferSize(int64_t indexSize)
{
    std::vector<int64_t> shapeVec = {indexSize};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, idType_, ge::DT_UINT32, false, config, maxValue, minValue);
    return static_cast<int64_t>(maxValue);
}

ge::graphStatus UnsortedSegmentSumSortSimtTiling::CalcTiling()
{
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(inputOuterDim_) + 1;
    int64_t mid = 0;
    int64_t sortTmpSize = 0;
    ubSize_ = ubSize_ - DCACHE_SIZE - SIMD_RESERVED_SIZE;
    while (end - start > 1) {
        mid = (end + start) / DOUBLE;
        int64_t totalIndexSize = Ops::Base::CeilAlign(mid * idTypeBytes_, ubBlockSize_) * DOUBLE +
                                 Ops::Base::CeilAlign(mid * idTypeBytes_, ubBlockSize_) + ubBlockSize_ * DOUBLE +
                                 Ops::Base::CeilAlign(mid * sizeof(uint32_t), ubBlockSize_);
        sortTmpSize = GetSortBufferSize(mid);
        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, static_cast<int64_t>(ubBlockSize_));
        int64_t tmpTotalSize =
            totalIndexSize + sortTmpSize + Ops::Base::CeilAlign(mid * innerDim_ * valueTypeBytes_, ubBlockSize_) * DOUBLE;
        if (tmpTotalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }
    int64_t rowUb = start;
    int64_t totalLoop = Ops::Base::CeilDiv(static_cast<int64_t>(inputOuterDim_), rowUb);
    int64_t eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<int64_t>(totalCoreNum_));
    usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
    int64_t tailCoreLoop = totalLoop - eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1);
    int64_t tailIndexNum = static_cast<int64_t>(inputOuterDim_) -
                           rowUb * eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1) - rowUb * (tailCoreLoop - 1);
    while (usedCoreNum_ < (totalCoreNum_ / static_cast<uint64_t>(DOUBLE)) && rowUb > 1) {
        rowUb = rowUb / DOUBLE;
        totalLoop = Ops::Base::CeilDiv(static_cast<int64_t>(inputOuterDim_), rowUb);
        eachCoreLoop = Ops::Base::CeilDiv(totalLoop, static_cast<int64_t>(totalCoreNum_));
        usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, eachCoreLoop);
        tailCoreLoop = totalLoop - eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1);
        tailIndexNum = static_cast<int64_t>(inputOuterDim_) -
                       rowUb * eachCoreLoop * (static_cast<int64_t>(usedCoreNum_) - 1) - rowUb * (tailCoreLoop - 1);
    }

    tilingData_.set_inputOuterDim(inputOuterDim_);
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_maxIndexNum(rowUb);
    tilingData_.set_oneCoreUbLoopTimes(eachCoreLoop);
    tilingData_.set_tailCoreUbLoopTimes(tailCoreLoop);
    tilingData_.set_maxThread(maxThread_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_sortTmpSize(sortTmpSize);
    tilingData_.set_tailIndexNum(tailIndexNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumSortSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SORT_SIMT;
    return tilingKey;
}

ge::graphStatus UnsortedSegmentSumSortSimtTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumSortSimtTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", inputOuterDim: " << tilingData_.get_inputOuterDim();
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", maxIndexNum: " << tilingData_.get_maxIndexNum();
    info << ", oneCoreUbLoopTimes: " << tilingData_.get_oneCoreUbLoopTimes();
    info << ", tailCoreUbLoopTimes: " << tilingData_.get_tailCoreUbLoopTimes();
    info << ", maxThread: " << tilingData_.get_maxThread();
    info << ", usedCoreNum: " << tilingData_.get_usedCoreNum();
    info << ", sortTmpSize: " << tilingData_.get_sortTmpSize();
    info << ", tailIndexNum: " << tilingData_.get_tailIndexNum();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumSortSimtTiling, 60);
}