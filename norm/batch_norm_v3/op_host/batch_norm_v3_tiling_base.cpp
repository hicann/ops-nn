/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_norm_v3_tiling_base.cpp
 * \brief
 */

#include "batch_norm_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"

static constexpr int64_t NCHW_DIM_NUM = 4;
static constexpr int64_t NCDHW_DIM_NUM = 5;
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t WEIGHT_INPUT_IDX = 1;
static constexpr int64_t BIAS_INPUT_IDX = 2;
static constexpr int64_t MEAN_INPUT_IDX = 3;
static constexpr int64_t VAR_INPUT_IDX = 4;
static constexpr int64_t EPS_ATTR_IDX = 0;
static constexpr int64_t MOMENTUM_ATTR_IDX = 1;
static constexpr int64_t IS_TRAINING_ATTR_IDX = 2;
static constexpr int64_t DIM_0 = 0;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;
static constexpr int64_t DIM_4 = 4;
static constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
static constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
static constexpr int64_t B32_BLOCK_ALIGN_NUM = 8;
static constexpr float DEFAULT_EPSILON = 1e-5;
static constexpr float DEFAULT_MOMENTUM = 0.1;

using namespace ge;

namespace optiling {
static inline bool IsDtypeSupported(const ge::DataType dtype)
{
    return ((dtype == ge::DT_FLOAT16) || (dtype == ge::DT_BF16) || (dtype == ge::DT_FLOAT));
}

bool BatchNormV3TilingBase::CheckInputDtype()
{
    OP_CHECK_IF(
        context_->GetInputDesc(X_INPUT_IDX) == nullptr, OP_LOGE("BatchNormV3TilingBase", "X_INPUT_IDX is null"),
        return false);
    auto xDtype = context_->GetInputDesc(X_INPUT_IDX)->GetDataType();
    OP_CHECK_IF(
        context_->GetInputDesc(WEIGHT_INPUT_IDX) == nullptr,
        OP_LOGE("BatchNormV3TilingBase", "WEIGHT_INPUT_IDX is null"), return false);
    auto weightDtype = context_->GetInputDesc(WEIGHT_INPUT_IDX)->GetDataType();
    OP_CHECK_IF(
        context_->GetInputDesc(BIAS_INPUT_IDX) == nullptr, OP_LOGE("BatchNormV3TilingBase", "BIAS_INPUT_IDX is null"),
        return false);
    auto biasDtype = context_->GetInputDesc(BIAS_INPUT_IDX)->GetDataType();
    OP_CHECK_IF(
        context_->GetInputDesc(MEAN_INPUT_IDX) == nullptr, OP_LOGE("BatchNormV3TilingBase", "MEAN_INPUT_IDX is null"),
        return false);
    auto meanDtype = context_->GetInputDesc(MEAN_INPUT_IDX)->GetDataType();
    OP_CHECK_IF(
        context_->GetInputDesc(VAR_INPUT_IDX) == nullptr, OP_LOGE("BatchNormV3TilingBase", "VAR_INPUT_IDX is null"),
        return false);

    auto varDtype = context_->GetInputDesc(VAR_INPUT_IDX)->GetDataType();
    OP_CHECK_IF(
        !IsDtypeSupported(xDtype), OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), "DT_FLOAT, DT_FLOAT16 or DT_BF16"),
        return false);
    OP_CHECK_IF(
        (!IsDtypeSupported(weightDtype)),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight",
            ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str(), "DT_FLOAT, DT_FLOAT16 or DT_BF16"), return false);
    OP_CHECK_IF(
        (xDtype != weightDtype) && (weightDtype != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "weight",
            ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str(),
            "when weight dtype not same as x dtype, weight dtype must be DT_FLOAT"),
        return false);
    if (weightDtype != biasDtype) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(weightDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(biasDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "weight and bias",
            dtypesStr.c_str(), "the datatype of bias and weight must be the same");
        return false;
    }
    OP_CHECK_IF(
        (meanDtype != ge::DT_FLOAT), OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "running_mean",
            ge::TypeUtils::DataTypeToSerialString(meanDtype).c_str(), "DT_FLOAT"),
        return false);
    OP_CHECK_IF(
        (varDtype != ge::DT_FLOAT), OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "running_var",
            ge::TypeUtils::DataTypeToSerialString(varDtype).c_str(), "DT_FLOAT"),
        return false);
    commonParams.xDtype = xDtype;
    return true;
}

bool BatchNormV3TilingBase::CheckInputShape()
{
    auto xShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE("BatchNormV3TilingBase", "xShape is null"), return false);
    auto xStorageShape = xShape->GetStorageShape();
    auto weightShape = context_->GetInputShape(WEIGHT_INPUT_IDX);
    OP_CHECK_IF(weightShape == nullptr, OP_LOGE("BatchNormV3TilingBase", "weightShape is null"), return false);
    auto bnWeightStorageShape = weightShape->GetStorageShape();
    auto bnBiasShape = context_->GetInputShape(BIAS_INPUT_IDX);
    OP_CHECK_IF(bnBiasShape == nullptr, OP_LOGE("BatchNormV3TilingBase", "biasShape is null"), return false);
    auto bnBiasStorageShape = bnBiasShape->GetStorageShape();
    auto bnMeanShape = context_->GetInputShape(MEAN_INPUT_IDX);
    OP_CHECK_IF(bnMeanShape == nullptr, OP_LOGE("BatchNormV3TilingBase", "meanShape is null"), return false);
    auto bnMeanStorageShape = bnMeanShape->GetStorageShape();
    auto bnVarShape = context_->GetInputShape(VAR_INPUT_IDX);
    OP_CHECK_IF(bnVarShape == nullptr, OP_LOGE("BatchNormV3TilingBase", "varShape is null"), return false);
    auto bnVarStorageShape = bnVarShape->GetStorageShape();
    auto xDesc = context_->GetInputDesc(X_INPUT_IDX);
    auto format = xDesc->GetFormat().GetStorageFormat();
    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                std::to_string(xStorageShape.GetDimNum()).c_str(), "4D with NCHW format"), return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCDHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                std::to_string(xStorageShape.GetDimNum()).c_str(), "5D with NCDHW format"), return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 =
            xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x",
            ge::TypeUtils::FormatToSerialString(format).c_str(), "NCHW and NCDHW");
        return false;
    }
    OP_CHECK_IF(
        commonParams.patternR1 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(xStorageShape).c_str(), "the dim 0 of x should be more than zero"),
        return false);
    OP_CHECK_IF(
        commonParams.patternA <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(xStorageShape).c_str(), "the dim 1 of x should be more than zero"),
        return false);
    OP_CHECK_IF(
        commonParams.patternR0 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(xStorageShape).c_str(), "dim_2 * dim_3 * dim_4 of x should be more than zero"),
        return false);
    if (bnWeightStorageShape.GetShapeSize() != commonParams.patternA) {
        std::string reasonMsg = "The shape size of weight must be the same as the C dimension of x (the second dimension is C dimension): " + std::to_string(commonParams.patternA);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "weight",
            std::to_string(bnWeightStorageShape.GetShapeSize()).c_str(), reasonMsg.c_str());
        return false;
    }
    if (bnBiasStorageShape.GetShapeSize() != commonParams.patternA) {
        std::string reasonMsg = "The shape size of bias must be the same as the C dimension of x (the second dimension is C dimension): " + std::to_string(commonParams.patternA);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "bias",
            std::to_string(bnBiasStorageShape.GetShapeSize()).c_str(), reasonMsg.c_str());
        return false;
    }
    if (bnMeanStorageShape.GetShapeSize() != commonParams.patternA) {
        std::string reasonMsg = "The shape size of running_mean must be the same as the C dimension of x (the second dimension is C dimension): " + std::to_string(commonParams.patternA);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "running_mean",
            std::to_string(bnMeanStorageShape.GetShapeSize()).c_str(), reasonMsg.c_str());
        return false;
    }
    if (bnVarStorageShape.GetShapeSize() != commonParams.patternA) {
        std::string reasonMsg = "The shape size of running_var must be the same as the C dimension of x (the second dimension is C dimension): " + std::to_string(commonParams.patternA);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "running_var",
            std::to_string(bnVarStorageShape.GetShapeSize()).c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

bool BatchNormV3TilingBase::CheckInputParam()
{
    OP_CHECK_IF(
        isTrainingValue == false,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "is_training",
            "false", "Attr is_training does not support false."),
        return false);
    // check输入dtype
    OP_CHECK_IF(
        !BatchNormV3TilingBase::CheckInputDtype(), OP_LOGE(commonParams.nodeName, "CheckInputDtype failed."),
        return false);
    // check输入shape
    OP_CHECK_IF(
        !BatchNormV3TilingBase::CheckInputShape(), OP_LOGE(commonParams.nodeName, "CheckInputShape failed."),
        return false);

    commonParams.patternR0Align = (commonParams.xDtype == ge::DT_FLOAT) ?
                                      Ops::Base::CeilAlign(commonParams.patternR0, B32_BLOCK_ALIGN_NUM) :
                                      Ops::Base::CeilAlign(commonParams.patternR0, B16_BLOCK_ALIGN_NUM);
    return true;
}

ge::graphStatus BatchNormV3TilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        commonParams.coreNum = ascendcPlatform.GetCoreNumAiv();
        socVersion = ascendcPlatform.GetSocVersion();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        commonParams.ubSizePlatForm = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const BatchNormV3CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(commonParams.nodeName, "compile info is null"), return ge::GRAPH_FAILED);
        commonParams.coreNum = compileInfoPtr->coreNum;
        commonParams.ubSizePlatForm = compileInfoPtr->ubSize;
    }
    OP_CHECK_IF(
        commonParams.coreNum == 0, OP_LOGE(commonParams.nodeName, "coreNum should not be equal to zero."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        commonParams.ubSizePlatForm == 0, OP_LOGE(commonParams.nodeName, "ubSizePlatForm should not be equal to zero."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3TilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("BatchNormV3", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const char* nodeNamePtr = context_->GetNodeName();
    commonParams.nodeName = (nodeNamePtr != nullptr) ? nodeNamePtr : "";
    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilonPtr = attrs->GetFloat(EPS_ATTR_IDX);
    commonParams.epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;
    const float* momentumPtr = attrs->GetFloat(MOMENTUM_ATTR_IDX);
    commonParams.momentum = (momentumPtr == nullptr) ? DEFAULT_MOMENTUM : *momentumPtr;
    const bool* isTrainingPtr = attrs->GetBool(IS_TRAINING_ATTR_IDX);
    isTrainingValue = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    commonParams.momentumReverse = 1 - commonParams.momentum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormV3TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
