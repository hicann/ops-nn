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
using namespace UnsortedSegmentSum;
static constexpr uint64_t LAST_DIM_SIMD_COND = 256;
static constexpr uint64_t BASE_A_SIZE = 1024;
static constexpr uint32_t DCACHE_SIZE = 32 * 1024;
static constexpr uint64_t DEFAULT_HANDLE_ROWS = 20UL;
static constexpr uint32_t NUMBER_THREE = 3;
static constexpr uint32_t DOUBLE = 2;
static constexpr uint32_t BIG_INNERDIM_THRESHOLD = 30;
static constexpr uint32_t HALF_UB_ALIGN = 16;
static constexpr uint64_t OFFSET_TYPESIZE = 8;              // int64_t offset for ScatterDataCopy
static constexpr uint64_t SCATTER_OFFSET_NUM_LIMIT = 65535; // uint16_t offsetNum upper bound
;

static const std::set<ge::DataType> determinsiticSupportType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool UnsortedSegmentSumDeterministicBigInnerDimTiling::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "[UnsortedSegmentSum] GetDeterministic state: %u", context_->GetDeterministic());
    if (context_->GetDeterministic() == 1 && dataShapeSize_ != 0 &&
        determinsiticSupportType.find(dataType_) != determinsiticSupportType.end() &&
        innerDim_ > BIG_INNERDIM_THRESHOLD) {
        // ascend960dt/ascend960pr 上 output fullload 模板优先级高于本模板(与 960 kernel 行为保持一致)，
        // 输出可全量驻留 UB 时让位给 output fullload 模板
        if (IsArch92Platform() && OutFlIsCapable()) {
            return false;
        }
        return true;
    }
    return false;
}

int64_t UnsortedSegmentSumDeterministicBigInnerDimTiling::FindMaxColsInUb()
{
    uint64_t localUbSize = ubSize_ - NUMBER_THREE * ubBlockSize_;
    int64_t maxCols = -1;
    int64_t sortTmpSize = GetSortTmpSize(baseS_);
    if (useScatterDMA_ == 0) {
        uint64_t oneColOccupyBytes = sizeof(float) * baseS_           // x
                                     + idTypeBytes_ * baseS_ * DOUBLE // segmentId、sortedId
                                     + sizeof(uint32_t) * baseS_      // sortedIdx
                                     + sizeof(float) * DOUBLE;        // y double buffer
        int64_t left = 0;
        int64_t right = static_cast<int64_t>(localUbSize) / static_cast<int64_t>(oneColOccupyBytes);
        while (left <= right) {
            int64_t mid = left + (right - left) / DOUBLE;
            int64_t occupyBytes = mid * static_cast<int64_t>(oneColOccupyBytes) + sortTmpSize;
            if (occupyBytes <= static_cast<int64_t>(localUbSize)) {
                maxCols = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return maxCols;
    }

    // ScatterDataCopy 路径的 UB 占用模型：
    // Per-column: xQueue (float, baseS_) + resBuf (float, baseS_) + resCastBuf (X_T, baseS_).
    // resCastBuf only exists for half/bf16; use sizeof(float) as a safe upper bound across dtypes.
    uint64_t perColBytesScatter = sizeof(float) * baseS_    // x
                                  + sizeof(float) * baseS_  // resBuf
                                  + sizeof(float) * baseS_; // resCastBuf (worst case = float size)
    // Fixed (column-independent): segmentId/sortedId/sortedIdx, dstSegIdBuf, offsetBuf, sortTmp.
    uint64_t fixedBytesCommon = idTypeBytes_ * baseS_ * DOUBLE                                   // segmentId、sortedId
                                + sizeof(uint32_t) * baseS_;                                     // sortedIdx
    uint64_t fixedBytesScatter = Ops::Base::CeilAlign(baseS_ * idTypeBytes_, ubBlockSize_)       // dstSegIdBuf
                                 + Ops::Base::CeilAlign(baseS_ * OFFSET_TYPESIZE, ubBlockSize_); // offsetBuf
    int64_t fixedTotal = static_cast<int64_t>(fixedBytesCommon + fixedBytesScatter) + sortTmpSize;
    int64_t left = 0;
    int64_t right = (static_cast<int64_t>(localUbSize) - fixedTotal) / static_cast<int64_t>(perColBytesScatter);
    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        int64_t occupyBytes = mid * static_cast<int64_t>(perColBytesScatter) + fixedTotal;
        if (occupyBytes <= static_cast<int64_t>(localUbSize)) {
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
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumDeterministicBigInnerDimTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->normalCoreProcessCols = normalCoreProcessCols_;
    tilingData->tailCoreProcessCols = tailCoreProcessCols_;
    tilingData->baseS = baseS_;
    tilingData->baseA = baseA_;
    tilingData->sortSharedBufSize = sortSharedBufSize_;
    tilingData->useScatterDMA = useScatterDMA_;
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
    useScatterDMA_ = IsArch92Platform() ? 1UL : 0UL;
    baseS_ = std::min(DEFAULT_HANDLE_ROWS, inputOuterDim_);
    if (useScatterDMA_ && baseS_ > SCATTER_OFFSET_NUM_LIMIT) {
        baseS_ = SCATTER_OFFSET_NUM_LIMIT;
    }
    int64_t colsNumInUB = FindMaxColsInUb();
    if (colsNumInUB <= 0) {
        OP_LOGE(context_->GetNodeName(),
                "One column data size is too large (innerDim is %lu), current module does not support.", innerDim_);
        return ge::GRAPH_PARAM_INVALID;
    }
    baseA_ = colsNumInUB / HALF_UB_ALIGN * HALF_UB_ALIGN;
    sortSharedBufSize_ = GetSortTmpSize(static_cast<int64_t>(baseS_));

    // ScatterDataCopy 路径按新的 UB 占用模型复核，超限时回退到常规路径并重算 baseA
    if (useScatterDMA_) {
        uint64_t perColBytesScatter = sizeof(float) * baseS_    // x
                                      + sizeof(float) * baseS_  // resBuf
                                      + sizeof(float) * baseS_; // resCastBuf (worst case)
        uint64_t fixedBytesCommon = idTypeBytes_ * baseS_ * DOUBLE + sizeof(uint32_t) * baseS_;
        uint64_t fixedBytesScatter = Ops::Base::CeilAlign(baseS_ * idTypeBytes_, ubBlockSize_) +
                                     Ops::Base::CeilAlign(baseS_ * OFFSET_TYPESIZE, ubBlockSize_);
        uint64_t totalCost = baseA_ * perColBytesScatter + fixedBytesCommon + fixedBytesScatter +
                             static_cast<uint64_t>(sortSharedBufSize_);
        uint64_t localUbSize = ubSize_ - NUMBER_THREE * ubBlockSize_;
        if (totalCost > localUbSize) {
            useScatterDMA_ = 0UL;
            int64_t legacyCols = FindMaxColsInUb();
            if (legacyCols <= 0) {
                OP_LOGE(context_->GetNodeName(),
                        "One column data size is too large (innerDim is %lu), current module does not support.",
                        innerDim_);
                return ge::GRAPH_PARAM_INVALID;
            }
            baseA_ = static_cast<uint64_t>(legacyCols) / HALF_UB_ALIGN * HALF_UB_ALIGN;
        }
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumDeterministicBigInnerDimTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_DETERMINISTIC_BIG_INNERDIM, USS_CAST_NONE);
    return tilingKey;
}

ge::graphStatus UnsortedSegmentSumDeterministicBigInnerDimTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumDeterministicBigInnerDimTiling::DumpTilingInfo()
{
    auto tilingData = context_->GetTilingData<UnsortedSegmentSumDeterministicBigInnerDimTilingData>();
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", normalCoreProcessCols: " << tilingData->normalCoreProcessCols;
    info << ", tailCoreProcessCols: " << tilingData->tailCoreProcessCols;
    info << ", baseS: " << tilingData->baseS;
    info << ", baseA: " << tilingData->baseA;
    info << ", sortSharedBufSize: " << tilingData->sortSharedBufSize;
    info << ", useScatterDMA: " << tilingData->useScatterDMA;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentSum", UnsortedSegmentSumDeterministicBigInnerDimTiling, 5);
} // namespace optiling
