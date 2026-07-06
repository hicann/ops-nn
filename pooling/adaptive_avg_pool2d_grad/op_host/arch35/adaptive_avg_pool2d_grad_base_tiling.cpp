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
 * \file adaptive_avg_pool2d_grad_base_tiling.cpp
 * \brief
 */
#include "adaptive_avg_pool2d_grad_base_tiling.h"
#include "error_util.h"
#include "op_host/tiling_util.h"

namespace optiling {
constexpr size_t CDHW_DIM_NUM = 4U;
constexpr uint64_t INPUT_GRAD_INDEX = 0;
constexpr size_t ORIGIN_INPUT_SHAPE = 0U;
constexpr uint64_t NCDHW_DIM_NUM = 5;
constexpr size_t NC_DIM_NUM = 2;
constexpr size_t C_DIM_OFFSET = 3; // pos = dim - offset
constexpr size_t H_DIM_OFFSET = 2;
constexpr size_t W_DIM_OFFSET = 1;
constexpr size_t INPUT_DIM_4D = 4;
constexpr size_t INPUT_DIM_3D = 3;

bool AdaptiveAvgPool2dGradTilingBase::CheckInputShape()
{
    const gert::StorageShape* inputGradShape = context_->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGradShape);
    const gert::Shape& inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputGradShape->GetStorageShape());
    size_t gradDimNum = inputShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto originInputShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ORIGIN_INPUT_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, originInputShapePtr);
    auto originDim = originInputShapePtr->GetSize();
    if (originDim != INPUT_DIM_4D && originDim != INPUT_DIM_3D) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "input dim", std::to_string(originDim),
                                     std::to_string(INPUT_DIM_4D) + " or " + std::to_string(INPUT_DIM_3D));
        return false;
    }
    auto originInputShapeSize = static_cast<const int64_t*>(originInputShapePtr->GetData());
    for (uint32_t i = 0; i < originDim; i++) {
        if (originInputShapeSize[i] == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "orig_input_shape",
                                                  std::to_string(originInputShapeSize[i]),
                                                  "The value is invalid for orig_input_shape");
            return false;
        }
    }
    const gert::StorageShape* outGradShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outGradShape);
    const gert::Shape& outputShape = Ops::NN::OpTiling::EnsureNotScalar(outGradShape->GetStorageShape());
    size_t outDimNum = outputShape.GetDimNum();

    if (originDim != gradDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "origin input dim", std::to_string(originDim),
                                     std::to_string(gradDimNum));
        return false;
    }
    if (originDim != outDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "origin input dim", std::to_string(originDim),
                                     std::to_string(outDimNum));
        return false;
    }
    for (uint32_t i = 0; i < originDim; i++) {
        if (originInputShapeSize[i] != outputShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "origin input shape",
                                          std::to_string(originInputShapeSize[i]),
                                          std::to_string(outputShape.GetDim(i)));
            return false;
        }
    }
    uint32_t loopSize = originDim == INPUT_DIM_4D ? NC_DIM_NUM : (NC_DIM_NUM - 1);

    for (uint32_t i = 0; i < loopSize; i++) {
        if (originInputShapeSize[i] != inputShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "origin input shape nc size",
                                          std::to_string(originInputShapeSize[i]),
                                          std::to_string(inputShape.GetDim(i)));
            return false;
        }
    }
    return true;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_GRAD_INDEX));
    auto gradDataType = context_->GetInputDesc(INPUT_GRAD_INDEX)->GetDataType();

    if ((gradDataType != ge::DT_FLOAT) && (gradDataType != ge::DT_FLOAT16) && (gradDataType != ge::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input data type",
                                  ge::TypeUtils::DataTypeToSerialString(gradDataType).c_str(),
                                  "float, float16, bfloat16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::SetInputParams()
{
    const gert::Shape gradShape = context_->GetInputShape(INPUT_GRAD_INDEX)->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto originInputShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ORIGIN_INPUT_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, originInputShapePtr);
    auto xDimNum = originInputShapePtr->GetSize();
    auto originInputShapeSize = static_cast<const int64_t*>(originInputShapePtr->GetData());

    uint32_t cPosIdx = xDimNum - C_DIM_OFFSET;
    uint32_t hPosIdx = xDimNum - H_DIM_OFFSET;
    uint32_t wPosIdx = xDimNum - W_DIM_OFFSET;

    inputData_.nX = (xDimNum == INPUT_DIM_3D) ? 1 : originInputShapeSize[0];
    inputData_.cX = originInputShapeSize[cPosIdx];
    inputData_.hX = originInputShapeSize[hPosIdx];
    inputData_.wX = originInputShapeSize[wPosIdx];
    inputData_.nGrad = (xDimNum == INPUT_DIM_3D) ? 1 : gradShape.GetDim(0);
    inputData_.cGrad = gradShape.GetDim(cPosIdx);
    inputData_.hGrad = gradShape.GetDim(hPosIdx);
    inputData_.wGrad = gradShape.GetDim(wPosIdx);
    inputData_.gradShapeSize = gradShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool2dGradTilingBase::IsGreaterThanInt32Max()
{
    int64_t cubeSize = inputData_.nX * inputData_.cX * inputData_.hX * inputData_.wX;
    return cubeSize > static_cast<int64_t>(INT32_MAX);
}

void AdaptiveAvgPool2dGradTilingBase::SetOtherInputParams()
{
    inputData_.inputDtype = context_->GetInputDesc(INPUT_GRAD_INDEX)->GetDataType();
    inputData_.isInt32Meet = IsGreaterThanInt32Max() ? 0 : 1;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveAvgPool2dGradTilingBase GetShapeAttrsInfo.");
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShape(), OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Get input shape failed."),
                return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool2dGradTilingBase::IsCapable() { return false; }

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = static_cast<const AdaptiveAvgPool2dGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize_ = static_cast<int64_t>(compileInfoPtr->ubSize);
        OP_LOGD(context_->GetNodeName(), "Get aivNum form compileInfo is: %ld, ubSize: %ld", coreNum_, ubSize_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        OP_LOGD(context_->GetNodeName(), "Get aivNum form ascendcPlatform is: %ld, ubSize: %ld", coreNum_, ubSize_);
    }
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            context_->GetNodeName(),
            "coreNum and ubSize should not be samller than 0, but got coreNum [%lu] and ubSize [%lu], please check.",
            coreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AdaptiveAvgPool2dGradTilingBase::GetTilingKey() const { return 0; }
} // namespace optiling
