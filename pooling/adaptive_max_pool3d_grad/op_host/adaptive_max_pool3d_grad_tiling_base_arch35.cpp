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
 * \file adaptive_max_pool3d_grad_tiling_base_arch35.cpp
 * \brief
 */

#include "error_util.h"
#include "adaptive_max_pool3d_grad_tiling_arch35.h"
#include "op_host/tiling_util.h"
#include "adaptive_max_pool3d_grad_tiling.h"

namespace optiling {

constexpr size_t CDHW_DIM_NUM = 4U;
constexpr uint64_t X_INDEX_V35 = 0;
constexpr uint64_t GRAD_INDEX_V35 = 1;
constexpr uint64_t ARGMAX_INDEX_V35 = 2;
constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;
constexpr int64_t nPos = 0;
constexpr int64_t cPos = 1;
constexpr int64_t dPos = 2;
constexpr int64_t hPos = 3;
constexpr int64_t wPos = 4;

bool AdaptiveMaxPool3dGradTilingBaseV35::CheckInputShape()
{
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX_V35);
    const gert::StorageShape* gradShape = context_->GetInputShape(GRAD_INDEX_V35);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gradDimNum = gradShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM),
                OP_LOGE(context_->GetNodeName(), "Input dim num should equal = %lu, actual is xDim: %lu, gradDim: %lu.",
                        NCDHW_DIM_NUM, xDimNum, gradDimNum),
                return false);

    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(xShape->GetStorageShape().GetDim(i) == 0,
                    OP_LOGE(context_->GetNodeName(), "Input x shape can not be 0."), return false);
    }

    for (size_t i = 0; i < NC_DIM_NUM; i++) {
        uint64_t xDimValue = xShape->GetStorageShape().GetDim(i);
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(
            (gradDimValue != xDimValue),
            OP_LOGE(context_->GetNodeName(), "Input dim check invalid, grad[%lu] is %lu, x[%lu] is %lu, not equal.", i,
                    gradDimValue, i, xDimValue),
            return false);
    }

    return true;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX_V35));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX_V35));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ARGMAX_INDEX_V35));
    auto xDataType = context_->GetInputDesc(X_INDEX_V35)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX_V35)->GetDataType();
    auto argmaxDataType = context_->GetInputDesc(ARGMAX_INDEX_V35)->GetDataType();

    OP_CHECK_IF(xDataType != gradDataType,
                OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not equal grad data type."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not fp32/fp16/bf16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((argmaxDataType != ge::DT_INT32) && (argmaxDataType != ge::DT_INT64),
                OP_LOGE(context_->GetNodeName(), "Data type invalid, argmax data type not int32/int64."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X_INDEX_V35)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX_V35)->GetStorageShape();

    inputData.nX = xShape.GetDim(nPos);
    inputData.cX = xShape.GetDim(cPos);
    inputData.dX = xShape.GetDim(dPos);
    inputData.hX = xShape.GetDim(hPos);
    inputData.wX = xShape.GetDim(wPos);
    inputData.nGrad = gradShape.GetDim(nPos);
    inputData.cGrad = gradShape.GetDim(cPos);
    inputData.dGrad = gradShape.GetDim(dPos);
    inputData.hGrad = gradShape.GetDim(hPos);
    inputData.wGrad = gradShape.GetDim(wPos);
    inputData.inputFormat = ge::Format::FORMAT_NCDHW;
    inputData.gradShapeSize = gradShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

static inline bool IsGreaterThanInt32Max(const AdaptiveMaxPool3dGradInputInfoV35& inputData)
{
    int64_t cubeSize = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    return cubeSize > static_cast<int64_t>(INT32_MAX);
}

void AdaptiveMaxPool3dGradTilingBaseV35::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(X_INDEX_V35)->GetDataType();
    inputData.argmaxDtype = context_->GetInputDesc(ARGMAX_INDEX_V35)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData) ? 0 : 1;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin 950, arch:%d.", npuArch);

    if (npuArch != NpuArch::DAV_3510) {
        return ge::GRAPH_PARAM_INVALID;
    }
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin.");

    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveMaxPool3dGradTilingBaseV35 GetShapeAttrsInfo.");
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShape(), OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(context_->GetNodeName(), "Set input shape failed."),
                return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3dGradTilingBaseV35::IsCapable() { return false; }

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const AdaptiveMaxPool3dGradCompileInfoV35*>(context_->GetCompileInfo());
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

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingBaseV35::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AdaptiveMaxPool3dGradTilingBaseV35::GetTilingKey() const { return 0; }
} // namespace optiling
