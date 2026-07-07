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
 * \file swiglu_group_quant_hifp8_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "swiglu_group_quant_hifp8_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t BATCH_MODE = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t SIZE_OF_FLOAT = 4;
constexpr int64_t UB_FACTOR_DYNAMIC = 7;
constexpr int64_t UB_FACTOR_STATIC = 6;

constexpr float CLAMP_LIMIT_DEFAULT = -1.0f;
constexpr float DST_TYPE_MAX_FINITE_DEFAULT = 15.0f;

constexpr int64_t QUANT_MODE_DYNAMIC = 3;
constexpr int64_t QUANT_MODE_STATIC = 2;

constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 4;
constexpr size_t ATTR_INDEX_DST_TYPE_MAX_FINITE = 5;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 6;

constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t INPUT_INDEX_SCALE = 3;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t OUTPUT_INDEX_Y_SCALE = 1;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 2;

constexpr int64_t DYNAMIC_HIFP8_TILING_KEY = 4000;
constexpr int64_t STATIC_HIFP8_TILING_KEY = 4100;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetPlatformInfo()
{
    return SwigluGroupQuantTiling::GetPlatformInfoCommon(context_, coreNum_, ubSize_);
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckInputDtype()
{
    auto xDtype = context_->GetInputDesc(INPUT_INDEX_X)->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
        OP_LOGE(context_->GetNodeName(), "input x dtype only support fp16/bf16/fp32, got %d.",
                  static_cast<int>(xDtype)),
        return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT),
            OP_LOGE(context_->GetNodeName(), "input weight dtype only support fp32, got %d.",
                      static_cast<int>(weightDtype)),
            return ge::GRAPH_FAILED);
    }

    auto groupDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupDesc != nullptr) {
        auto groupDtype = groupDesc->GetDataType();
        OP_CHECK_IF((groupDtype != ge::DT_INT64),
            OP_LOGE(context_->GetNodeName(), "group_index dtype only support int64, got %d.",
                      static_cast<int>(groupDtype)),
            return ge::GRAPH_FAILED);
    }

    if (quantMode_ == QUANT_MODE_STATIC) {
        auto scaleDesc = context_->GetOptionalInputDesc(INPUT_INDEX_SCALE);
        OP_CHECK_IF((scaleDesc == nullptr),
            OP_LOGE(context_->GetNodeName(), "scale input is required for static quant mode 2."),
            return ge::GRAPH_FAILED);
        auto scaleDtype = scaleDesc->GetDataType();
        OP_CHECK_IF((scaleDtype != ge::DT_FLOAT),
            OP_LOGE(context_->GetNodeName(), "scale dtype only support fp32, got %d.",
                      static_cast<int>(scaleDtype)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckOutputDtype()
{
    auto yDtype = context_->GetOutputDesc(OUTPUT_INDEX_Y)->GetDataType();
    OP_CHECK_IF((yDtype != ge::DT_HIFLOAT8),
        OP_LOGE(context_->GetNodeName(), "y dtype must be hifloat8, got %d.",
                  static_cast<int>(yDtype)),
        return ge::GRAPH_FAILED);

    if (quantMode_ == QUANT_MODE_DYNAMIC) {
        auto yScaleDtype = context_->GetOutputDesc(OUTPUT_INDEX_Y_SCALE)->GetDataType();
        OP_CHECK_IF((yScaleDtype != ge::DT_FLOAT),
            OP_LOGE(context_->GetNodeName(), "y_scale dtype must be fp32, got %d.",
                      static_cast<int>(yScaleDtype)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    if (quantModePtr != nullptr) {
        quantMode_ = *quantModePtr;
        OP_CHECK_IF((quantMode_ != QUANT_MODE_DYNAMIC && quantMode_ != QUANT_MODE_STATIC),
            OP_LOGE(context_->GetNodeName(), "quant_mode must be %ld (Dynamic) or %ld (Static), got %ld.",
                      QUANT_MODE_DYNAMIC, QUANT_MODE_STATIC, quantMode_),
            return ge::GRAPH_FAILED);
    }

    auto clampLimitPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    clampLimit_ = (clampLimitPtr != nullptr) ? *clampLimitPtr : CLAMP_LIMIT_DEFAULT;
    OP_CHECK_IF((clampLimit_ < 0.0f && clampLimit_ != -1.0f),
        OP_LOGE(context_->GetNodeName(), "clamp_limit must be -1 or > 0.0, got %f.", clampLimit_),
        return ge::GRAPH_FAILED);
    hasClamp_ = (clampLimit_ > 0.0f);

    auto dstTypeMaxPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_DST_TYPE_MAX_FINITE);
    dstTypeMax_ = (dstTypeMaxPtr != nullptr) ? *dstTypeMaxPtr : DST_TYPE_MAX_FINITE_DEFAULT;
    OP_CHECK_IF((dstTypeMax_ <= 0.0f),
        OP_LOGE(context_->GetNodeName(), "dst_type_max must be > 0, got %f.", dstTypeMax_),
        return ge::GRAPH_FAILED);

    auto outputOriginPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginPtr != nullptr) {
        outputOrigin_ = *outputOriginPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckWeightInfo()
{
    auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
    hasWeight_ = (weightShape != nullptr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckGroupIndexInfo()
{
    auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexShape == nullptr) {
        isGroup_ = false;
        groupNum_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    isGroup_ = true;
    auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
    size_t giDimNum = groupIndexStorageShape.GetDimNum();
    OP_CHECK_IF((giDimNum != 1),
        OP_LOGE(context_->GetNodeName(), "group_index must be 1D, got %zu dims.", giDimNum),
        return ge::GRAPH_FAILED);
    groupNum_ = groupIndexStorageShape.GetDim(0);
    OP_CHECK_IF((groupNum_ <= 0),
        OP_LOGE(context_->GetNodeName(), "group_index length must be > 0, got %ld.", groupNum_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckScaleInfo()
{
    if (quantMode_ != QUANT_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleShape = context_->GetOptionalInputShape(INPUT_INDEX_SCALE);
    OP_CHECK_IF((scaleShape == nullptr),
        OP_LOGE(context_->GetNodeName(), "scale input is required for quant_mode=2."),
        return ge::GRAPH_FAILED);
    auto scaleStorageShape = scaleShape->GetStorageShape();
    size_t scaleDimNum = scaleStorageShape.GetDimNum();
    OP_CHECK_IF((scaleDimNum != 1),
        OP_LOGE(context_->GetNodeName(), "scale must be 1D, got %zu dims.", scaleDimNum),
        return ge::GRAPH_FAILED);
    int64_t scaleSize = scaleStorageShape.GetDim(0);
    if (isGroup_) {
        OP_CHECK_IF((scaleSize != groupNum_),
            OP_LOGE(context_->GetNodeName(), "scale size [%ld] must equal group_index size [%ld] when group_index is present.",
                      scaleSize, groupNum_),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((scaleSize != 1),
            OP_LOGE(context_->GetNodeName(), "scale size [%ld] must be 1 when group_index is not present.", scaleSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYShape(size_t xDimNum, const gert::Shape& xStorageShape)
{
    auto yShape = context_->GetOutputShape(OUTPUT_INDEX_Y);
    OP_CHECK_IF((yShape == nullptr),
        OP_LOGE(context_->GetNodeName(), "y shape is null."),
        return ge::GRAPH_FAILED);
    const gert::Shape &yShapeStorage = yShape->GetStorageShape();
    OP_CHECK_IF((yShapeStorage.GetDimNum() != xDimNum),
        OP_LOGE(context_->GetNodeName(), "y dim num [%zu] must equal x dim num [%zu].",
                  yShapeStorage.GetDimNum(), xDimNum),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum - 1; i++) {
        OP_CHECK_IF((yShapeStorage.GetDim(i) != xStorageShape.GetDim(i)),
            OP_LOGE(context_->GetNodeName(), "y dim[%zu] [%ld] must equal x dim[%zu] [%ld].",
                      i, yShapeStorage.GetDim(i), i, xStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((yShapeStorage.GetDim(xDimNum - 1) != dimH_),
        OP_LOGE(context_->GetNodeName(), "y last dim [%ld] must equal dimH [%ld].",
                  yShapeStorage.GetDim(xDimNum - 1), dimH_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYScaleShape()
{
    if (quantMode_ == QUANT_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    auto yScaleShape = context_->GetOutputShape(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_IF((yScaleShape == nullptr),
        OP_LOGE(context_->GetNodeName(), "y_scale shape is null."),
        return ge::GRAPH_FAILED);
    const gert::Shape &yScaleShapeStorage = yScaleShape->GetStorageShape();
    OP_CHECK_IF((yScaleShapeStorage.GetDimNum() != 1),
        OP_LOGE(context_->GetNodeName(), "y_scale must be 1D, got %zu dims.", yScaleShapeStorage.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (isGroup_) {
        OP_CHECK_IF((yScaleShapeStorage.GetDim(0) != groupNum_),
            OP_LOGE(context_->GetNodeName(), "y_scale dim[0] [%ld] must equal groupNum [%ld] in group mode.",
                      yScaleShapeStorage.GetDim(0), groupNum_),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((yScaleShapeStorage.GetDim(0) != 1),
            OP_LOGE(context_->GetNodeName(), "y_scale dim[0] [%ld] must be 1 in non-group mode.",
                      yScaleShapeStorage.GetDim(0)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYOriginShape(size_t xDimNum, const gert::Shape& xStorageShape)
{
    if (!outputOrigin_) {
        return ge::GRAPH_SUCCESS;
    }
    auto yOriginShape = context_->GetOutputShape(OUTPUT_INDEX_Y_ORIGIN);
    OP_CHECK_IF((yOriginShape == nullptr),
        OP_LOGE(context_->GetNodeName(), "y_origin shape is null when output_origin=true."),
        return ge::GRAPH_FAILED);
    const gert::Shape &yOriginShapeStorage = yOriginShape->GetStorageShape();
    OP_CHECK_IF((yOriginShapeStorage.GetDimNum() != xDimNum),
        OP_LOGE(context_->GetNodeName(), "y_origin dim num [%zu] must equal x dim num [%zu].",
                  yOriginShapeStorage.GetDimNum(), xDimNum),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum - 1; i++) {
        OP_CHECK_IF((yOriginShapeStorage.GetDim(i) != xStorageShape.GetDim(i)),
            OP_LOGE(context_->GetNodeName(), "y_origin dim[%zu] [%ld] must equal x dim[%zu] [%ld].",
                      i, yOriginShapeStorage.GetDim(i), i, xStorageShape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((yOriginShapeStorage.GetDim(xDimNum - 1) != dimH_),
        OP_LOGE(context_->GetNodeName(), "y_origin last dim [%ld] must equal dimH [%ld].",
                  yOriginShapeStorage.GetDim(xDimNum - 1), dimH_),
        return ge::GRAPH_FAILED);
    auto yOriginDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    auto xDtype = context_->GetInputDesc(INPUT_INDEX_X)->GetDataType();
    if (yOriginDesc != nullptr) {
        OP_CHECK_IF((yOriginDesc->GetDataType() != xDtype),
            OP_LOGE(context_->GetNodeName(), "y_origin dtype must equal x dtype."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckOutputInfo()
{
    auto xShape = context_->GetInputShape(INPUT_INDEX_X);
    auto xStorageShape = xShape->GetStorageShape();
    size_t xDimNum = xStorageShape.GetDimNum();

    if (CheckYShape(xDimNum, xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYScaleShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYOriginShape(xDimNum, xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetShapeAttrsInfoInner()
{
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);

    size_t xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF((xDimNum < 1),
        OP_LOGE(context_->GetNodeName(), "x dims must be >= 1, got %zu.", xDimNum),
        return ge::GRAPH_FAILED);
    totalTokens_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        totalTokens_ *= xStorageShape.GetDim(i);
    }
    dim2H_ = xStorageShape.GetDim(xDimNum - 1);
    OP_CHECK_IF((dim2H_ % SWI_FACTOR != 0),
        OP_LOGE(context_->GetNodeName(), "x last dim must be even, got %ld.", dim2H_),
        return ge::GRAPH_FAILED);
    dimH_ = dim2H_ / SWI_FACTOR;

    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantHifp8Tiling::CalcTileTokens()
{
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - UB_RESERVE;
    int64_t ubFactor = (quantMode_ == QUANT_MODE_STATIC) ? UB_FACTOR_STATIC : UB_FACTOR_DYNAMIC;
    int64_t ubPerToken = ubFactor * dimH_ * SIZE_OF_FLOAT;
    if (hasWeight_) {
        ubPerToken += DB_BUFFER * SIZE_OF_FLOAT;
    }

    tileTokens_ = ubAvailable / ubPerToken;
    if (tileTokens_ < 1) {
        tileTokens_ = 1;
    }
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CalcCoreDistribution()
{
    if (!isGroup_) {
        usedCoreNum_ = std::min(static_cast<int64_t>(coreNum_), totalTokens_);
        if (usedCoreNum_ == 0) {
            return ge::GRAPH_FAILED;
        }
        tokensPerCore_ = totalTokens_ / usedCoreNum_;
    } else {
        usedCoreNum_ = std::min({static_cast<int64_t>(coreNum_), groupNum_, MAX_CORE_COUNT});
        tokensPerCore_ = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CalcOpTiling()
{
    CalcTileTokens();
    tileLength_ = tileTokens_ * dimH_;
    return CalcCoreDistribution();
}

void SwigluGroupQuantHifp8Tiling::SetTilingData()
{
    tilingData_.set_totalTokens(totalTokens_);
    tilingData_.set_dim2H(dim2H_);
    tilingData_.set_dimH(dimH_);
    tilingData_.set_isGroup(isGroup_ ? 1 : 0);
    tilingData_.set_hasWeight(hasWeight_ ? 1 : 0);
    tilingData_.set_hasClamp(hasClamp_ ? 1 : 0);
    tilingData_.set_outputOrigin(outputOrigin_ ? 1 : 0);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_dstTypeMax(dstTypeMax_);
    tilingData_.set_tileTokens(tileTokens_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_tokensPerCore(tokensPerCore_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_tileLength(tileLength_);
}

void SwigluGroupQuantHifp8Tiling::SetTilingKey()
{
    tilingKey_ = (quantMode_ == QUANT_MODE_STATIC) ? STATIC_HIFP8_TILING_KEY : DYNAMIC_HIFP8_TILING_KEY;
    context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        workspaceSize_ = 0;
        if (!isGroup_ && quantMode_ != QUANT_MODE_STATIC) {
            workspaceSize_ += (usedCoreNum_ + 1) * SIZE_OF_FLOAT + BLOCK_SIZE;
        }
        return ge::GRAPH_SUCCESS;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize_ = sysWorkspaceSize;
    if (!isGroup_ && quantMode_ != QUANT_MODE_STATIC) {
        workspaceSize_ += (usedCoreNum_ + 1) * SIZE_OF_FLOAT + BLOCK_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::PostTiling()
{
    context_->SetScheduleMode(BATCH_MODE);
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckAllInputs()
{
    if (CheckInputDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetAttr() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckWeightInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::ProcessTiling()
{
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAllInputs() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ProcessTiling();
}

}  // namespace optiling
