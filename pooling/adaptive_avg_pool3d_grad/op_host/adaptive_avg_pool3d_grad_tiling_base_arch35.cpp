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
constexpr size_t C_DIM_OFFSET = 4;  // pos = dim - offset
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

    // data_format should be NCDHW or NDHWC
    OP_CHECK_IF(!(data_formatStr == "NCDHW" || data_formatStr == "NDHWC"),
                OP_LOGE(context_->GetNodeName(), "ATTR data_format is %s ,expect [NDHWC] or [NCDHW].", data_format),
                return false);
    
    // xDimNum should be 5 or 4
    OP_CHECK_IF(((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM)) &&
                ((xDimNum != CDHW_DIM_NUM) || (gradDimNum != CDHW_DIM_NUM)),
                OP_LOGE(context_->GetNodeName(),
                        "Input dim num should equal = %lu or %lu, actual is xDim: %lu, gradDim: %lu",
                        NCDHW_DIM_NUM, CDHW_DIM_NUM, xDimNum, gradDimNum),
                return false);
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(xShape->GetStorageShape().GetDim(i) == 0,
                    OP_LOGE(context_->GetNodeName(), "Input x shape can not be 0."), return false);
    }

    // Input NCDim should be equal
    uint32_t cPosIdx = (data_formatStr == "NDHWC") ? xDimNum - 1 : xDimNum - 4;
    uint64_t xNDim = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (gradDimNum == CDHW_DIM_NUM) ? 1 : gradShape->GetStorageShape().GetDim(0);
    uint64_t xCDim = xShape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = gradShape->GetStorageShape().GetDim(cPosIdx);
    OP_CHECK_IF((xNDim != gradNDim) || (xCDim != gradCDim),
                OP_LOGE(context_->GetNodeName(), "Input N,C dim check invalid, grad(%lu,%lu), x(%lu,%lu), not equal.",
                        gradNDim, gradCDim, xNDim, xCDim),
                return false);
    return true;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    
    OP_CHECK_IF(
        xDataType != gradDataType,
        OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not equal to grad data type."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
        OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not fp32/fp16/bf16."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::SetInputParams()
{
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    std::string data_formatStr = data_format;

    uint32_t cPosIdx = xDimNum - C_DIM_OFFSET; 
    uint32_t dPosIdx = xDimNum - D_DIM_OFFSET; 
    uint32_t hPosIdx = xDimNum - H_DIM_OFFSET; 
    uint32_t wPosIdx = xDimNum - W_DIM_OFFSET; 
    
    inputData.inputFormat = ge::Format::FORMAT_NCDHW;

    if (data_formatStr == "NDHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NDHWC;
        dPosIdx = dPosIdx - 1; 
        hPosIdx = hPosIdx - 1;
        wPosIdx = wPosIdx - 1;
        cPosIdx = xDimNum - 1;
    }

    inputData.nX = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape.GetDim(0);
    inputData.cX = xShape.GetDim(cPosIdx);
    inputData.dX = xShape.GetDim(dPosIdx);
    inputData.hX = xShape.GetDim(hPosIdx);
    inputData.wX = xShape.GetDim(wPosIdx);
    inputData.nGrad = (xDimNum == CDHW_DIM_NUM) ? 1 : gradShape.GetDim(0);
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

bool AdaptiveAvgPool3dGradTilingBaseV35::IsCapable()
{
    return false;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const AdaptiveAvgPool3dGradCompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus AdaptiveAvgPool3dGradTilingBaseV35::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool3dGradTilingBaseV35::GetTilingKey() const
{
    return 0;
}
} // namespace optiling
