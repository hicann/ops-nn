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
 * \file adaptive_max_pool3d_tiling.cpp
 * \brief
 */

#include "adaptive_max_pool3d_tiling.h"

using Ops::NN::Optiling::TilingRegistry;

namespace optiling {
constexpr uint64_t NCDHW_DIM_N = 0;
constexpr uint64_t NCDHW_DIM_C = 1;
constexpr uint64_t NCDHW_DIM_D = 2;
constexpr uint64_t NCDHW_DIM_H = 3;
constexpr uint64_t NCDHW_DIM_W = 4;
constexpr uint64_t OUTPUTSIZE_DIMW = 2;
constexpr uint64_t OUTPUTSIZE_DIM_MAX = 3;
constexpr uint64_t DIM_NUM_FIVE = 5;
constexpr uint64_t ASCENDC_WORKSPACE = 16 * 1024 * 1024;
static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}

bool AdaptiveMaxPool3dTilingBase::IsCapable()
{
    return true;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveMaxPool3dTilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<AdaptiveMaxPool3dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    input_.coreNum = compileInfo->coreNum;
    input_.ubSizePlatForm = compileInfo->ubSizePlatForm;
    OP_CHECK_IF(input_.coreNum <= 0, OP_LOGE(context_, "GetPlatformInfo get corenum <= 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetShapeAttrsInfo()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin.");

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
        OP_LOGE(nodeName, "x datatype only support float, float16, bfloat16"), return ge::GRAPH_FAILED);
    input_.xDtype = xDtype;
    gert::Shape xShape = EnsureNotScalar(inputX->GetStorageShape());
    if (xShape.GetDimNum() == DIM_NUM_FIVE) {
        input_.N = xShape.GetDim(NCDHW_DIM_N);
        input_.C = xShape.GetDim(NCDHW_DIM_C);
        input_.Di = xShape.GetDim(NCDHW_DIM_D);
        input_.Hi = xShape.GetDim(NCDHW_DIM_H);
        input_.Wi = xShape.GetDim(NCDHW_DIM_W);
    } else {
        OP_LOGE(nodeName, "xShape dim number should be 5");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        input_.N < 1 || input_.C < 1 || input_.Di < 1 || input_.Hi < 1 || input_.Wi < 1,
        OP_LOGE(nodeName, "Invalid shape. Maybe empty tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.Di * input_.Hi * input_.Wi > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        OP_LOGE(nodeName, "no support for D*H*W of input greater than int32 max value"), return ge::GRAPH_FAILED);

    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSizePtr);
    OP_CHECK_IF(
        outputSizePtr->GetSize() != OUTPUTSIZE_DIM_MAX, OP_LOGE(nodeName, "the size of outputsize only support 3"),
        return ge::GRAPH_FAILED);
    const int64_t* outputSize = static_cast<const int64_t*>(outputSizePtr->GetData());
    OP_CHECK_IF(
        outputSize[0] <= 0 || outputSize[1] <= 0 || outputSize[OUTPUTSIZE_DIMW] <= 0,
        OP_LOGE(nodeName, "the value of outputsize should > 0"), return ge::GRAPH_FAILED);
    input_.Do = outputSize[0];
    input_.Ho = outputSize[1];
    input_.Wo = outputSize[OUTPUTSIZE_DIMW];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = ASCENDC_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdaptiveMaxPool3d(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4AdaptiveMaxPool3d(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4AdaptiveMaxPool3d enter.");

    auto compileInfo = context->GetCompiledInfo<AdaptiveMaxPool3dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;

    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get corenum size"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "TilingPrepare4AdaptiveMaxPool3d end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveMaxPool3d)
    .Tiling(Tiling4AdaptiveMaxPool3d)
    .TilingParse<AdaptiveMaxPool3dCompileInfo>(TilingPrepare4AdaptiveMaxPool3d);
} // namespace optiling
