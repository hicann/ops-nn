/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling_base.cpp
 * \brief SwigluGroupGrad shared tiling validation (arch35, Ascend950)
 */

#include "swiglu_group_grad_tiling_base.h"
#include <string>
#include <limits>
#include <graph/utils/type_utils.h>

namespace optiling {

ge::graphStatus GetSwigluGroupGradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    } else {
        auto compileInfoPtr = context->GetCompileInfo<SwigluGroupGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNum = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize = compileInfoPtr->ubSize;
    }

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcDtype(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    auto inputDesc = context->GetInputDesc(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    inputData.gradYDtype = inputDesc->GetDataType();

    if (inputData.gradYDtype != ge::DT_BF16 && inputData.gradYDtype != ge::DT_FLOAT16 &&
        inputData.gradYDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "grad_y dtype[%s] not supported",
                ge::TypeUtils::DataTypeToSerialString(inputData.gradYDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    auto gradYStorageShape = context->GetInputShape(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYStorageShape);
    const gert::Shape& gradYShape = gradYStorageShape->GetStorageShape();

    OP_CHECK_IF(gradYShape.GetDimNum() < 1,
                OP_LOGE(context->GetNodeName(), "grad_y must be at least 1D, got %ld dims.", gradYShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    inputData.totalRows = 1;
    for (size_t i = 0; i < gradYShape.GetDimNum() - 1; ++i) {
        inputData.totalRows *= gradYShape.GetDim(i);
    }

    inputData.H = gradYShape.GetDim(gradYShape.GetDimNum() - 1);

    auto xStorageShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();

    OP_CHECK_IF(xShape.GetDimNum() < 1,
                OP_LOGE(context->GetNodeName(), "x must be at least 1D, got %ld dims.", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    int64_t xTotalRows = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        xTotalRows *= xShape.GetDim(i);
    }
    inputData.dim2H = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_CHECK_IF(xTotalRows != inputData.totalRows,
                OP_LOGE(context->GetNodeName(), "x rows=%ld != grad_y rows=%ld", xTotalRows, inputData.totalRows),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.dim2H != inputData.H * DIM_TWO,
                OP_LOGE(context->GetNodeName(), "x.shape[-1]=%ld != 2*H=%ld", inputData.dim2H, inputData.H * DIM_TWO),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.H <= 0, OP_LOGE(context->GetNodeName(), "H=%ld must be > 0", inputData.H),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseOptionalInputs(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    inputData.isWeight = 0;
    inputData.isYOrigin = 0;
    inputData.isGroupIndex = 0;
    inputData.groupIndexG = 0;

    const gert::Shape* weightShape = nullptr;
    const gert::Shape* yOriginShape = nullptr;
    const gert::Shape* groupIndexShape = nullptr;
    int64_t floatOptCount = 0;
    for (int64_t i = WEIGHT_INDEX; i <= GROUP_INDEX_INDEX; i++) {
        auto shapePtr = context->GetInputShape(i);
        if (shapePtr == nullptr) {
            continue;
        }
        const gert::Shape& optShape = shapePtr->GetStorageShape();
        if (optShape.GetDimNum() == 0) {
            continue;
        }
        auto desc = context->GetInputDesc(i);
        if (desc == nullptr) {
            continue;
        }
        if (desc->GetDataType() == ge::DT_INT64) {
            groupIndexShape = &optShape;
        } else {
            floatOptCount++;
            if (floatOptCount == 1) {
                weightShape = &optShape;
            } else {
                yOriginShape = &optShape;
            }
        }
    }

    if (weightShape != nullptr) {
        inputData.isWeight = 1;
        auto gradWeightOut = context->GetOutputShape(1);
        OP_CHECK_IF(gradWeightOut == nullptr,
                    OP_LOGE(context->GetNodeName(), "grad_weight must be non-null when weight is present"),
                    return ge::GRAPH_FAILED);
        const gert::Shape& gradWeightShape = gradWeightOut->GetStorageShape();
        OP_CHECK_IF(gradWeightShape.GetDimNum() != weightShape->GetDimNum(),
                    OP_LOGE(context->GetNodeName(), "grad_weight dims=%ld must match weight dims=%ld",
                            gradWeightShape.GetDimNum(), weightShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(weightShape->GetDimNum()); ++i) {
            OP_CHECK_IF(gradWeightShape.GetDim(i) != weightShape->GetDim(i),
                        OP_LOGE(context->GetNodeName(), "grad_weight.shape[%ld]=%ld must match weight.shape[%ld]=%ld",
                                i, gradWeightShape.GetDim(i), i, weightShape->GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(yOriginShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "y_origin must be present when weight is present"),
                    return ge::GRAPH_FAILED);
        inputData.isYOrigin = 1;

        OP_CHECK_IF(yOriginShape->GetDimNum() < 1, OP_LOGE(context->GetNodeName(), "y_origin must be at least 1D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(yOriginShape->GetDim(yOriginShape->GetDimNum() - 1) != inputData.H,
                    OP_LOGE(context->GetNodeName(), "y_origin.shape[-1]=%ld must equal H=%ld",
                            yOriginShape->GetDim(yOriginShape->GetDimNum() - 1), inputData.H),
                    return ge::GRAPH_FAILED);
        int64_t yOriginTotalRows = 1;
        for (size_t i = 0; i < yOriginShape->GetDimNum() - 1; ++i) {
            yOriginTotalRows *= yOriginShape->GetDim(i);
        }
        OP_CHECK_IF(yOriginTotalRows != inputData.totalRows,
                    OP_LOGE(context->GetNodeName(), "y_origin outer numel(%ld) must equal totalRows(%ld)",
                            yOriginTotalRows, inputData.totalRows),
                    return ge::GRAPH_FAILED);

        auto weightElementNum = weightShape->GetShapeSize();
        if (weightElementNum != inputData.totalRows) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "weight", std::to_string(weightElementNum).c_str(),
                "The element num of weight must be equal to the product of grad_y leading dims.");
            return ge::GRAPH_FAILED;
        }
    }

    if (groupIndexShape != nullptr) {
        inputData.isGroupIndex = 1;
        OP_CHECK_IF(groupIndexShape->GetDimNum() != 1 || groupIndexShape->GetDim(0) < 1,
                    OP_LOGE(context->GetNodeName(), "group_index must be a non-empty 1D tensor, got dimNum=%ld.",
                            groupIndexShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        inputData.groupIndexG = groupIndexShape->GetDim(0);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseAttrs(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    inputData.hasClamp = 0;
    inputData.clampLimit = 0.0f;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* clampLimitAttr = attrs->GetAttrPointer<float>(CLAMPLIMIT_ATTR_INDEX);

    if (clampLimitAttr != nullptr) {
        OP_CHECK_IF(*clampLimitAttr != -1.0f && !(*clampLimitAttr > 0.0f),
                    OP_LOGE(context->GetNodeName(), "clamp_limit must be -1.0 (no clamp) or > 0.0, but got %f",
                            *clampLimitAttr),
                    return ge::GRAPH_FAILED);
        if (*clampLimitAttr > 0.0f) {
            inputData.hasClamp = 1;
            inputData.clampLimit = *clampLimitAttr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSwigluGroupGradShapeAttrsInfo(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    OP_CHECK_IF(CalcDtype(context, inputData) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CalcDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseOptionalInputs(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ParseOptionalInputs failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAttrs(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ParseAttrs failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
