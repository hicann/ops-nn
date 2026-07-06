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
 * \file adaptive_avg_pool3d_grad_tiling_base_arch35.cpp
 * \brief
 */

#include "error_util.h"
#include "adaptive_avg_pool3d_grad_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "adaptive_avg_pool3d_grad_tiling.h"

namespace optiling {

constexpr size_t CDHW_DIM_NUM = 4U;
constexpr uint64_t GRAD_INDEX = 0;
constexpr uint64_t X_INDEX = 1;
constexpr size_t DATA_FORMAT_ATTR_INDEX = 0U;
constexpr uint64_t NCDHW_DIM_NUM = 5;
constexpr size_t NC_DIM_NUM = 2;
constexpr size_t C_DIM_OFFSET = 4; // pos = dim - offset
constexpr size_t D_DIM_OFFSET = 3;
constexpr size_t H_DIM_OFFSET = 2;
constexpr size_t W_DIM_OFFSET = 1;
constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;

bool AdaptiveAvgPool3dGradTilingBaseV35::CheckInputShape()
{
    const gert::StorageShape* gradShape = context_->GetInputShape(GRAD_INDEX);
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);

    size_t gradDimNum = gradShape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    std::string data_formatStr = data_format;

    // data_format should be NCDHW or NDHWC or CDHW or DHWC
    OP_CHECK_IF(!(data_formatStr == "NCDHW" || data_formatStr == "NDHWC" || data_formatStr == "CDHW" ||
                  data_formatStr == "DHWC"),
                OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "data_format", data_formatStr.c_str(),
                                           "NCDHW, NDHWC, CDHW, or DHWC"),
                return false);

    // xDimNum should be 5 or 4, and gradDimNum should be the same rank as xDimNum.
    OP_CHECK_IF(
        ((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM)) &&
            ((xDimNum != CDHW_DIM_NUM) || (gradDimNum != CDHW_DIM_NUM)),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x, grad",
            ("xDim=" + std::to_string(xDimNum) + ", gradDim=" + std::to_string(gradDimNum)).c_str(),
            ("the dims of x and grad must be " + std::to_string(NCDHW_DIM_NUM) + " or " + std::to_string(CDHW_DIM_NUM))
                .c_str()),
        return false);

    for (uint32_t i = 0; i < xDimNum; i++) {
        uint64_t xDimVal = xShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(xDimVal == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimVal).c_str(),
                        ("the dim" + std::to_string(i) + " of x should not be 0").c_str()),
                    return false);
    }

    for (uint32_t i = 0; i < gradDimNum; i++) {
        uint64_t gDimVal = gradShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(gDimVal == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "grad", std::to_string(gDimVal).c_str(),
                        ("the dim" + std::to_string(i) + " of grad should not be 0").c_str()),
                    return false);
    }

    uint32_t cPosIdx = (data_formatStr == "NCDHW" || data_formatStr == "CDHW") ?
                           static_cast<uint32_t>(xDimNum - C_DIM_OFFSET) :
                           static_cast<uint32_t>(xDimNum - W_DIM_OFFSET);

    uint64_t xNDim = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (gradDimNum == CDHW_DIM_NUM) ? 1 : gradShape->GetStorageShape().GetDim(0);
    uint64_t xCDim = xShape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = gradShape->GetStorageShape().GetDim(cPosIdx);

    OP_CHECK_IF(
        (xNDim != gradNDim) || (xCDim != gradCDim),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad, x",
                                               ("N=" + std::to_string(gradNDim) + "," + std::to_string(gradCDim) +
                                                " and N=" + std::to_string(xNDim) + "," + std::to_string(xCDim))
                                                   .c_str(),
                                               "the N,C dims of grad and x must be equal"),
        return false);

    return true;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));

    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();

    OP_CHECK_IF(xDataType != gradDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x, grad",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDataType) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradDataType))
                                                           .c_str(),
                                                       "the dtypes of x and grad must be the same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDataType).c_str(),
                                                      "the dtype of x must be fp32, fp16, or bf16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::SetInputParams()
{
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();

    size_t xDimNum = xShape.GetDimNum();
    size_t gradDimNum = gradShape.GetDimNum();

    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    std::string data_formatStr = data_format;

    uint32_t cPosIdx = 0;
    uint32_t dPosIdx = 0;
    uint32_t hPosIdx = 0;
    uint32_t wPosIdx = 0;

    inputData.inputFormat = ge::Format::FORMAT_NCDHW;

    if (data_formatStr == "NCDHW" || data_formatStr == "CDHW") {
        // NCDHW: [N, C, D, H, W]
        // CDHW : [C, D, H, W]
        cPosIdx = static_cast<uint32_t>(xDimNum - C_DIM_OFFSET);
        dPosIdx = static_cast<uint32_t>(xDimNum - D_DIM_OFFSET);
        hPosIdx = static_cast<uint32_t>(xDimNum - H_DIM_OFFSET);
        wPosIdx = static_cast<uint32_t>(xDimNum - W_DIM_OFFSET);
        inputData.inputFormat = ge::Format::FORMAT_NCDHW;
    } else {
        // NDHWC: [N, D, H, W, C]
        // DHWC : [D, H, W, C]
        dPosIdx = static_cast<uint32_t>(xDimNum - C_DIM_OFFSET);
        hPosIdx = static_cast<uint32_t>(xDimNum - D_DIM_OFFSET);
        wPosIdx = static_cast<uint32_t>(xDimNum - H_DIM_OFFSET);
        cPosIdx = static_cast<uint32_t>(xDimNum - W_DIM_OFFSET);
        inputData.inputFormat = ge::Format::FORMAT_NDHWC;
    }

    inputData.nX = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape.GetDim(0);
    inputData.cX = xShape.GetDim(cPosIdx);
    inputData.dX = xShape.GetDim(dPosIdx);
    inputData.hX = xShape.GetDim(hPosIdx);
    inputData.wX = xShape.GetDim(wPosIdx);

    inputData.nGrad = (gradDimNum == CDHW_DIM_NUM) ? 1 : gradShape.GetDim(0);
    inputData.cGrad = gradShape.GetDim(cPosIdx);
    inputData.dGrad = gradShape.GetDim(dPosIdx);
    inputData.hGrad = gradShape.GetDim(hPosIdx);
    inputData.wGrad = gradShape.GetDim(wPosIdx);

    inputData.gradShapeSize = gradShape.GetShapeSize();

    return ge::GRAPH_SUCCESS;
}

static inline bool IsGreaterThanInt32Max(const AdaptiveAvgPool3dGradInputInfo& inputData)
{
    int64_t cubeSize = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    return cubeSize > static_cast<int64_t>(INT32_MAX);
}

void AdaptiveAvgPool3dGradTilingBaseV35::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData) ? 0 : 1;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        // Skip the current template
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveAvgPool3dGradTilingBaseV35 GetShapeAttrsInfo.");

    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckInputShape(), OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(context_->GetNodeName(), "Set input shape failed."),
                return ge::GRAPH_FAILED);

    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool3dGradTilingBaseV35::IsCapable() { return false; }

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const AdaptiveAvgPool3dGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSizePlatForm;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
    }

    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AdaptiveAvgPool3dGradTilingBaseV35::GetTilingKey() const { return 0; }

} // namespace optiling