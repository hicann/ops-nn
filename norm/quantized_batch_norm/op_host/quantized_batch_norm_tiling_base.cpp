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
 * \file quantized_batch_norm_tiling_base.cc
 * \brief
 */

#include <iostream>
#include <string>
#include "error_util.h"
#include "quantized_batch_norm_tiling.h"

namespace {
static constexpr int64_t NCHW_DIM_NUM = 4;
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t MEAN_INPUT_IDX = 1;
static constexpr int64_t VAR_INPUT_IDX = 2;
static constexpr int64_t INSCALE_INPUT_IDX = 3;
static constexpr int64_t INZP_INPUT_IDX = 4;
static constexpr int64_t OUTSCALE_INPUT_IDX = 5;
static constexpr int64_t OUTZP_INPUT_IDX = 6;
static constexpr int64_t WEIGHT_INPUT_IDX = 7;
static constexpr int64_t BIAS_INPUT_IDX = 8;
static constexpr int64_t EPS_ATTR_IDX = 0;
static constexpr int64_t DIM_0 = 0;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;
static constexpr int64_t DIM_4 = 4;
static constexpr uint32_t MINIMAL_WORKSPACE = 16777216; // 16 * 1024 * 1024
static constexpr int64_t I8_BLOCK_ALIGN_NUM = 32;
static constexpr int64_t I32_BLOCK_ALIGN_NUM = 8;
}; // namespace

namespace optiling {
static inline bool IsDtypeSupported(const ge::DataType dtype)
{
    return ((dtype == ge::DT_INT8) || (dtype == ge::DT_UINT8) || (dtype == ge::DT_INT32));
}

static inline bool IsOthersDtypeSupported(const ge::DataType dtype)
{
    return (dtype == ge::DT_FLOAT);
}

bool QuantizedBatchNormTilingBase::CheckInputDtype()
{
    OP_CHECK_IF(
        context_->GetInputDesc(X_INPUT_IDX) == nullptr, OP_LOGE(commonParams.nodeName, "Get input X desc failed"),
        return false);
    auto xDtype = context_->GetInputDesc(X_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(WEIGHT_INPUT_IDX) == nullptr,
        OP_LOGE(commonParams.nodeName, "Get input weight desc failed"), return false);
    auto weightDtype = context_->GetInputDesc(WEIGHT_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(BIAS_INPUT_IDX) == nullptr, OP_LOGE(commonParams.nodeName, "Get input bias desc failed"),
        return false);
    auto biasDtype = context_->GetInputDesc(BIAS_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(MEAN_INPUT_IDX) == nullptr, OP_LOGE(commonParams.nodeName, "Get input mean desc failed"),
        return false);
    auto meanDtype = context_->GetInputDesc(MEAN_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(VAR_INPUT_IDX) == nullptr, OP_LOGE(commonParams.nodeName, "Get input var desc failed"),
        return false);
    auto varDtype = context_->GetInputDesc(VAR_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(INSCALE_INPUT_IDX) == nullptr,
        OP_LOGE(commonParams.nodeName, "Get input inScale desc failed"), return false);
    auto inScaleDtype = context_->GetInputDesc(INSCALE_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(INZP_INPUT_IDX) == nullptr,
        OP_LOGE(commonParams.nodeName, "Get input inZeroPoint desc failed"), return false);
    auto inZeroPointDtype = context_->GetInputDesc(INZP_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(OUTSCALE_INPUT_IDX) == nullptr,
        OP_LOGE(commonParams.nodeName, "Get input outScale desc failed"), return false);
    auto outScaleDtype = context_->GetInputDesc(OUTSCALE_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        context_->GetInputDesc(OUTZP_INPUT_IDX) == nullptr,
        OP_LOGE(commonParams.nodeName, "Get input outZeroPoint desc failed"), return false);
    auto outZeroPointDtype = context_->GetInputDesc(OUTZP_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(
        !IsDtypeSupported(xDtype), OP_LOGE(commonParams.nodeName, "x dtype must in [DT_INT8, DT_UINT8, DT_INT32]"),
        return false);

    OP_CHECK_IF(
        (!IsOthersDtypeSupported(weightDtype)), OP_LOGE(commonParams.nodeName, "weight dtype must in [DT_FLOAT]"),
        return false);

    OP_CHECK_IF(
        (weightDtype != biasDtype), OP_LOGE(commonParams.nodeName, "bias dtype must be same as weight dtype"),
        return false);

    OP_CHECK_IF(
        (meanDtype != ge::DT_FLOAT), OP_LOGE(commonParams.nodeName, "running_mean dtype should be DT_FLOAT"),
        return false);

    OP_CHECK_IF(
        (varDtype != ge::DT_FLOAT), OP_LOGE(commonParams.nodeName, "running_var dtype should be DT_FLOAT"),
        return false);

    OP_CHECK_IF(
        (inScaleDtype != ge::DT_FLOAT), OP_LOGE(commonParams.nodeName, "input_scale dtype should be DT_FLOAT "),
        return false);

    OP_CHECK_IF(
        (inZeroPointDtype != ge::DT_INT32),
        OP_LOGE(commonParams.nodeName, "input_zero_point dtype should be DT_INT32 "), return false);

    OP_CHECK_IF(
        (outScaleDtype != ge::DT_FLOAT), OP_LOGE(commonParams.nodeName, "output_scale dtype should be DT_FLOAT"),
        return false);

    OP_CHECK_IF(
        (outZeroPointDtype != ge::DT_INT32),
        OP_LOGE(commonParams.nodeName, "output_zero_point dtype should be DT_INT32"), return false);
    commonParams.xDtype = xDtype;
    return true;
}

bool QuantizedBatchNormTilingBase::CheckInputShape()
{
    auto xShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input X shape failed"), return false);
    auto xStorageShape = xShape->GetStorageShape();

    auto weightShape = context_->GetInputShape(WEIGHT_INPUT_IDX);
    OP_CHECK_IF(weightShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input weight shape failed"), return false);
    auto weightStorageShape = weightShape->GetStorageShape();

    auto biasShape = context_->GetInputShape(BIAS_INPUT_IDX);
    OP_CHECK_IF(biasShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input bias shape failed"), return false);
    auto biasStorageShape = biasShape->GetStorageShape();

    auto meanShape = context_->GetInputShape(MEAN_INPUT_IDX);
    OP_CHECK_IF(meanShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input mean shape failed"), return false);
    auto meanStorageShape = meanShape->GetStorageShape();

    auto xDesc = context_->GetInputDesc(X_INPUT_IDX);
    auto format = xDesc->GetFormat().GetStorageFormat();
    if (format == ge::FORMAT_NCHW) {
        OP_CHECK_IF(
            xStorageShape.GetDimNum() != NCHW_DIM_NUM, OP_LOGE(commonParams.nodeName, "Not supported x format"),
            return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else {
        OP_LOGE(commonParams.nodeName, "Not supported x format");
        return false;
    }

    OP_CHECK_IF(
        commonParams.patternR1 <= 0, OP_LOGE(commonParams.nodeName, "x shape dim 0 should be more than zero."),
        return false);
    OP_CHECK_IF(
        commonParams.patternA <= 0, OP_LOGE(commonParams.nodeName, "x shape dim 1 should be more than zero."),
        return false);
    OP_CHECK_IF(
        commonParams.patternR0 <= 0, OP_LOGE(commonParams.nodeName, "x shape dim_2 * dim_3 should be more than zero."),
        return false);
    OP_CHECK_IF(
        weightStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE(
            commonParams.nodeName, "weight ShapeSize: %ld should equal x shape C dim: %ld",
            weightStorageShape.GetShapeSize(), commonParams.patternA),
        return false);
    OP_CHECK_IF(
        biasStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE(
            commonParams.nodeName, "bias ShapeSize: %ld should equal x shape C dim: %ld",
            biasStorageShape.GetShapeSize(), commonParams.patternA),
        return false);
    OP_CHECK_IF(
        meanStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE(
            commonParams.nodeName, "mean ShapeSize: %ld should equal x shape C dim: %ld",
            meanStorageShape.GetShapeSize(), commonParams.patternA),
        return false);
    return true;
}

bool QuantizedBatchNormTilingBase::CheckOthersInputShape()
{
    auto varShape = context_->GetInputShape(VAR_INPUT_IDX);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input var shape failed"), return false);
    auto varStorageShape = varShape->GetStorageShape();

    auto inScaleShape = context_->GetInputShape(INSCALE_INPUT_IDX);
    OP_CHECK_IF(
        inScaleShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input inScale shape failed"), return false);
    auto inScaleStorageShape = inScaleShape->GetStorageShape();

    auto inZeroPointShape = context_->GetInputShape(INZP_INPUT_IDX);
    OP_CHECK_IF(
        inZeroPointShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input inZeroPoint shape failed"),
        return false);
    auto inZeroPointStorageShape = inZeroPointShape->GetStorageShape();

    auto outScaleShape = context_->GetInputShape(OUTSCALE_INPUT_IDX);
    OP_CHECK_IF(
        outScaleShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input outScale shape failed"), return false);
    auto outScaleStorageShape = outScaleShape->GetStorageShape();

    auto outZeroPointShape = context_->GetInputShape(OUTZP_INPUT_IDX);
    OP_CHECK_IF(
        outZeroPointShape == nullptr, OP_LOGE(commonParams.nodeName, "Get input outZeroPoint shape failed"),
        return false);
    auto outZeroPointStorageShape = outZeroPointShape->GetStorageShape();

    OP_CHECK_IF(
        varStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE(
            commonParams.nodeName, "var ShapeSize: %ld should equal x shape C dim: %ld", varStorageShape.GetShapeSize(),
            commonParams.patternA),
        return false);

    OP_CHECK_IF(
        inScaleStorageShape.GetShapeSize() != 1,
        OP_LOGE(commonParams.nodeName, "input_scale ShapeSize: %ld should equal 1", inScaleStorageShape.GetShapeSize()),
        return false);
    OP_CHECK_IF(
        inZeroPointStorageShape.GetShapeSize() != 1,
        OP_LOGE(
            commonParams.nodeName, "input_zero_point ShapeSize: %ld should equal 1",
            inZeroPointStorageShape.GetShapeSize()),
        return false);
    OP_CHECK_IF(
        outScaleStorageShape.GetShapeSize() != 1,
        OP_LOGE(
            commonParams.nodeName, "output_scale ShapeSize: %ld should equal 1", outScaleStorageShape.GetShapeSize()),
        return false);
    OP_CHECK_IF(
        outZeroPointStorageShape.GetShapeSize() != 1,
        OP_LOGE(
            commonParams.nodeName, "output_zero_point ShapeSize: %ld should equal 1",
            outZeroPointStorageShape.GetShapeSize()),
        return false);
    return true;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        commonParams.coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        commonParams.ubSizePlatForm = ubSizePlatForm;
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<QuantizedBatchNormCompileInfo>();
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(commonParams.nodeName, "compile info is null"), return ge::GRAPH_FAILED);
        commonParams.coreNum = compileInfoPtr->coreNum;
        commonParams.ubSizePlatForm = compileInfoPtr->ubSize;
    }
    OP_CHECK_IF(
        commonParams.coreNum == 0, OP_LOGE(commonParams.nodeName, "numBlocks should not be equal to zero."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        commonParams.ubSizePlatForm == 0, OP_LOGE(commonParams.nodeName, "ubSize should not be equal to zero."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("QuantizedBatchNorm", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    commonParams.nodeName = context_->GetNodeName();
    // 获取attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(EPS_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsilon);
    commonParams.epsilon = *epsilon;
    // check输入dtype
    OP_CHECK_IF(
        !QuantizedBatchNormTilingBase::CheckInputDtype(), OP_LOGE("GetShapeAttrsInfo", "CheckInputDtype failed."),
        return ge::GRAPH_FAILED);
    // check输入shape
    OP_CHECK_IF(
        !QuantizedBatchNormTilingBase::CheckInputShape(), OP_LOGE("GetShapeAttrsInfo", "CheckInputShape failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !QuantizedBatchNormTilingBase::CheckOthersInputShape(), OP_LOGE("GetShapeAttrsInfo", "CheckInputShape failed."),
        return ge::GRAPH_FAILED);
    commonParams.patternR0Align = (commonParams.xDtype == ge::DT_INT32) ?
                                      Ops::Base::CeilAlign(commonParams.patternR0, I32_BLOCK_ALIGN_NUM) :
                                      Ops::Base::CeilAlign(commonParams.patternR0, I8_BLOCK_ALIGN_NUM);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling