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
 * \file repeat_interleave_tiling_normal.cpp
 * \brief
 */

#include "repeat_interleave_tiling_normal.h"

namespace optiling {
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t ALIGN_COUNT = 32;
static constexpr int64_t MIN_CP_THRESHOLD = 128;
static constexpr int64_t MIN_SHAPE_THRESHOLD = 64;
static constexpr int64_t BATCH_TILING_REPEAT_SCALAR = 101;
static constexpr int64_t BATCH_TILING_REPEAT_TENSOR = 102;
static constexpr int64_t BATCH_TILING_SPLIT_REPEATS_SHAPE = 201;
static constexpr int64_t MIN_TOTAL_REPEATS_SUM = 1024;
static constexpr int64_t REDUCE_SUM_BUFFER = 32;

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

bool RepeatInterleaveTilingKernelNorm::IsCapable()
{
    return true;
}
void RepeatInterleaveTilingKernelNorm::SplitRepeatsShape()
{
    int64_t ratio = totalCoreNum_ / usedCoreNumBefore_;
    int64_t repeatsShape = repeatShape_.GetDim(0);
    normalRepeatsCount_ = CeilDiv(repeatsShape, ratio);
    repeatsSlice_ = CeilDiv(repeatsShape, normalRepeatsCount_);
    usedCoreNum_ = usedCoreNumBefore_ * repeatsSlice_;
    tailRepeatsCount_ = repeatsShape - normalRepeatsCount_ * (repeatsSlice_ - 1);
    isSplitShape_ = true;
    return;
}
void RepeatInterleaveTilingKernelNorm::SplitCP()
{
    int64_t ratio = totalCoreNum_ / usedCoreNum_;
    int64_t cpAxis = 2;
    int64_t cpDim = mergedDim_[cpAxis];
    int64_t cpMin = MIN_CP_THRESHOLD / ge::GetSizeByDataType(inputDtype_);
    normalCP_ = std::max(CeilDiv(cpDim, ratio), cpMin);
    cpSlice_ = CeilDiv(cpDim, normalCP_);
    usedCoreNum_ = usedCoreNum_ * cpSlice_;
    tailCP_ = cpDim - normalCP_ * (cpSlice_ - 1);
    isSplitCP_ = 1;
    return;
}
void RepeatInterleaveTilingKernelNorm::GetUbFactor()
{
    int64_t batchNum = mergedDim_[0];
    eachCoreBatchCount_ = CeilDiv(batchNum, totalCoreNum_);
    usedCoreNum_ = CeilDiv(batchNum, eachCoreBatchCount_);
    usedCoreNumBefore_ = usedCoreNum_;
    tailCoreBatchCount_ = batchNum - eachCoreBatchCount_ * (usedCoreNum_ - 1);
    totalRepeatSum_ = yShape_.GetDim(axis_);
    // 核数不满一半，继续切分CP轴
    int64_t cpAxis = 2;
    if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
        (mergedDim_[cpAxis] * ge::GetSizeByDataType(inputDtype_) > MIN_CP_THRESHOLD)) {
        SplitCP();
    }
    // CP轴切分后，核数还是不满，则切分repeats shape
    if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
        (repeatShape_.GetDim(0) > MIN_SHAPE_THRESHOLD || (mergedDim_[cpAxis] == 1 && repeatShape_.GetDim(0) != 1)) &&
        (totalRepeatSum_ > MIN_TOTAL_REPEATS_SUM)) {
        SplitRepeatsShape();
    }

    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        repeatsCount_ = yShape_.GetDim(axis_) / inputShape_.GetDim(axis_);
    }
    averageRepeatTime_ = yShape_.GetDim(axis_) / inputShape_.GetDim(axis_);
    int64_t ubFactor = 0;
    if (isSplitShape_) {
        ubSize_ -= (totalCoreNum_ * ALIGN_COUNT * DOUBLE);
        ubFactor = (ubSize_ - REDUCE_SUM_BUFFER * ge::GetSizeByDataType(repeatDtype_)) /
                   (ge::GetSizeByDataType(inputDtype_) * DOUBLE + ge::GetSizeByDataType(repeatDtype_));
    } else {
        ubFactor =
            ubSize_ / DOUBLE / (ge::GetSizeByDataType(inputDtype_) * DOUBLE + ge::GetSizeByDataType(repeatDtype_));
    }
    ubFactor_ = ubFactor / ALIGN_COUNT * ALIGN_COUNT;
    return;
}
ge::graphStatus RepeatInterleaveTilingKernelNorm::DoOpTiling()
{
    MergDim();
    GetUbFactor();
    return ge::GRAPH_SUCCESS;
}

uint64_t RepeatInterleaveTilingKernelNorm::GetTilingKey() const
{
    if (isSplitShape_) {
        return BATCH_TILING_SPLIT_REPEATS_SHAPE;
    }
    if (repeatsCount_ > -1) {
        return BATCH_TILING_REPEAT_SCALAR;
    }
    return BATCH_TILING_REPEAT_TENSOR;
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    if (isSplitShape_) {
        kernelTilingDataSmall_.set_totalCoreNum(totalCoreNum_);
        kernelTilingDataSmall_.set_usedCoreNum(usedCoreNum_);
        kernelTilingDataSmall_.set_eachCoreBatchCount(eachCoreBatchCount_);
        kernelTilingDataSmall_.set_tailCoreBatchCount(tailCoreBatchCount_);
        kernelTilingDataSmall_.set_normalRepeatsCount(normalRepeatsCount_);
        kernelTilingDataSmall_.set_tailRepeatsCount(tailRepeatsCount_);
        kernelTilingDataSmall_.set_repeatsSlice(repeatsSlice_);
        kernelTilingDataSmall_.set_ubFactor(ubFactor_);
        kernelTilingDataSmall_.set_totalRepeatSum(totalRepeatSum_);
        kernelTilingDataSmall_.set_averageRepeatTime(averageRepeatTime_);
        kernelTilingDataSmall_.set_mergedDims(mergedDim_);
        kernelTilingDataSmall_.SaveToBuffer(
            context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(kernelTilingDataSmall_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    kernelTilingData_.set_totalCoreNum(totalCoreNum_);
    kernelTilingData_.set_usedCoreNum(usedCoreNum_);
    kernelTilingData_.set_eachCoreBatchCount(eachCoreBatchCount_);
    kernelTilingData_.set_tailCoreBatchCount(tailCoreBatchCount_);
    kernelTilingData_.set_isSplitCP(isSplitCP_);
    kernelTilingData_.set_normalCP(normalCP_);
    kernelTilingData_.set_tailCP(tailCP_);
    kernelTilingData_.set_cpSlice(cpSlice_);
    kernelTilingData_.set_ubFactor(ubFactor_);
    kernelTilingData_.set_repeatsCount(repeatsCount_);
    kernelTilingData_.set_totalRepeatSum(totalRepeatSum_);
    kernelTilingData_.set_mergedDims(mergedDim_);

    kernelTilingData_.SaveToBuffer(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(kernelTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void RepeatInterleaveTilingKernelNorm::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalCoreNum: " << totalCoreNum_;
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", eachCoreBatchCount: " << eachCoreBatchCount_;
    info << ", tailCoreBatchCount: " << tailCoreBatchCount_;
    info << ", isSplitCP: " << isSplitCP_;
    info << ", normalCP: " << normalCP_;
    info << ", tailCP: " << tailCP_;
    info << ", cpSlice: " << cpSlice_;
    info << ", ubFactor: " << ubFactor_;
    info << ", repeatsCount: " << repeatsCount_;
    info << ", totalRepeatSum: " << totalRepeatSum_;
    info << ", isSplitShape: " << isSplitShape_;
    info << ", normalRepeatsCount: " << normalRepeatsCount_;
    info << ", tailRepeatsCount: " << tailRepeatsCount_;
    info << ", repeatsSlice: " << repeatsSlice_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
    return;
}
} // namespace optiling
