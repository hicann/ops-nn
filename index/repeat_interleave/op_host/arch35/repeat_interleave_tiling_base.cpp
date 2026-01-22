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
 * \file repeat_interleave_tiling_base.cpp
 * \brief
 */

#include "repeat_interleave_tiling_base.h"
#include "repeat_interleave_tiling_arch35.h"

namespace optiling {
static constexpr uint32_t INDEX_X_INPUT = 0;
static constexpr uint32_t INDEX_REPEATS_INPUT = 1;
static constexpr uint32_t AXIS_DEFAULT_VALUE = 1000;
static constexpr uint32_t CACHELINE_SIZE = 128;
static constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const std::set<ge::DataType> SUPPORTED_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_INT8,
                                                       ge::DT_BOOL,  ge::DT_BF16,    ge::DT_INT16, ge::DT_UINT16,
                                                       ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64};

bool RepeatInterleaveBaseTiling::IsCapable()
{
    return true;
}
ge::graphStatus RepeatInterleaveBaseTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const RepeatInterleaveCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNumAscendc;
    ubSize_ = compileInfo->ubSizeAscendc;
    return ge::GRAPH_SUCCESS;
}

int64_t RepeatInterleaveBaseTiling::MergeDimExceptAxis(const gert::Shape& input, int64_t axis)
{
    int64_t result = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(input.GetDimNum()); i++) {
        if (i != axis) {
            result *= input.GetDim(i);
        }
    }
    return result;
}

ge::graphStatus RepeatInterleaveBaseTiling::CheckDtype()
{
    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(
        !IsSupportDtype(SUPPORTED_DTYPE, inputDtype_),
        OP_LOGE(
            context_->GetNodeName(),
            "The dtype only support float, float16, bfloat16, \
        int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool currently, please check."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputDtype_ != yDtype, OP_LOGE(context_->GetNodeName(), "The x and y must have the same dtype, please check."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveBaseTiling::CheckShape()
{
    int64_t inputDimMerge = MergeDimExceptAxis(inputShape_, axis_);
    int64_t outputDimMerge = MergeDimExceptAxis(yShape_, axis_);

    OP_CHECK_IF(
        (!isDefaultAxis_) && (axis_ < 0 || axis_ >= static_cast<int64_t>(inputShape_.GetDimNum())),
        OP_LOGE(context_->GetNodeName(), "The attribute axis is invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        inputShape_.GetDim(axis_) == 0,
        OP_LOGE(context_->GetNodeName(), "The input x shape of axis dim must be nonzero."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        repeatShape_.GetDimNum() > 1,
        OP_LOGE(context_->GetNodeName(), "The input repeats should be 1D tensor or a scalar."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (repeatShape_.GetDimNum() != 0 && repeatShape_.GetDim(0) != 1) &&
            (repeatShape_.GetDim(0) != inputShape_.GetDim(axis_)),
        OP_LOGE(
            context_->GetNodeName(),
            "The input repeats shape should be 1 or a scalar or the same as input axis shape."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (!isDefaultAxis_) && (inputShape_.GetDimNum() != yShape_.GetDimNum()),
        OP_LOGE(context_->GetNodeName(), "The input dim and output dim should be the same when axis isn't none."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (!isDefaultAxis_) && (inputDimMerge != outputDimMerge),
        OP_LOGE(context_->GetNodeName(), "The output shape is invalid when axis isn't none."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveBaseTiling::GetShapeAttrsInfo()
{
    auto xShapePtr = context_->GetRequiredInputShape(INDEX_X_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    inputShape_ = xShapePtr->GetStorageShape();

    auto repeatShapePtr = context_->GetRequiredInputShape(INDEX_REPEATS_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, repeatShapePtr);
    repeatShape_ = repeatShapePtr->GetStorageShape();

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    yShape_ = yShapePtr->GetStorageShape();

    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrAxis = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    axis_ = *attrAxis;

    axis_ = axis_ < 0 ? axis_ + inputShape_.GetDimNum() : axis_;
    if (axis_ == AXIS_DEFAULT_VALUE) {
        axis_ = inputShape_.GetDimNum() - 1;
        isDefaultAxis_ = true;
    }

    auto inputXPtr = context_->GetRequiredInputDesc(INDEX_X_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    inputDtype_ = inputXPtr->GetDataType();

    auto repeatsPtr = context_->GetRequiredInputDesc(INDEX_REPEATS_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, repeatsPtr);
    repeatDtype_ = repeatsPtr->GetDataType();

    ge::graphStatus ret = CheckShape();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Check shape failed.");
        return ret;
    }
    ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Check dtype failed.");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RepeatInterleaveBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus RepeatInterleaveBaseTiling::GetWorkspaceSize()
{
    auto sysWorkspace = SYS_WORKSPACE_SIZE;
    sysWorkspace += totalCoreNum_ * CACHELINE_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

void RepeatInterleaveBaseTiling::MergDimForTensor()
{
    int64_t n = 1;
    int64_t cp = 1;
    for (uint32_t i = 0; i < axis_; i++) {
        n *= inputShape_.GetDim(i);
    }

    for (uint32_t i = axis_ + 1; i < inputShape_.GetDimNum(); i++) {
        cp *= inputShape_.GetDim(i);
    }
    mergedDim_[0] = n;
    mergedDim_[1] = inputShape_.GetDim(axis_);
    mergedDim_[REPEAT_INTERLEAVE_MERGED_DIM_LENGTH - 1] = cp;
    return;
}

void RepeatInterleaveBaseTiling::MergDimForScalar()
{
    int64_t n = 1;
    int64_t cp = 1;
    for (uint32_t i = 0; i <= axis_; i++) {
        n *= inputShape_.GetDim(i);
    }
    for (uint32_t i = axis_ + 1; i < inputShape_.GetDimNum(); i++) {
        cp *= inputShape_.GetDim(i);
    }
    mergedDim_[0] = n;
    mergedDim_[1] = 1;
    mergedDim_[REPEAT_INTERLEAVE_MERGED_DIM_LENGTH - 1] = cp;
    return;
}

void RepeatInterleaveBaseTiling::MergDim()
{
    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        MergDimForScalar();
    } else {
        MergDimForTensor();
    }
    return;
}
} // namespace optiling