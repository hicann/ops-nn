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
 * \file apply_gradient_descent_tiling_arch35.cpp
 * \brief apply_gradient_descent_tiling
 */

#include "apply_gradient_descent_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
//#include "util/bfloat16.h"
#include "op_host/tiling_util.h"
#include "../../op_kernel/arch35/apply_gradient_descent_dag.h"
#include "../../op_kernel/arch35/apply_gradient_descent_tiling_key.h"

#include <iostream>
namespace optiling {
using namespace ApplyGradientDescentOp;
const size_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024
const gert::Shape g_vec_1_shape = {1};

ge::graphStatus ApplyGradientDescentTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTiling SetTilingData enter.");
    schMode_ = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode_);
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTilingData : tilingKey is %lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyGradientDescentTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTiling CalcInputDtype enter.");
    auto varDesc = tilingContext->GetInputDesc(0);
    auto alphaDesc = tilingContext->GetInputDesc(1);
    auto deltaDesc = tilingContext->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, varDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, alphaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, deltaDesc);
    this->varDtype = varDesc->GetDataType();
    this->alphaDtype = alphaDesc->GetDataType();
    this->deltaDtype = deltaDesc->GetDataType();

    OP_CHECK_IF(
        this->varDtype != ge::DT_FLOAT && this->varDtype != ge::DT_FLOAT16 && this->varDtype != ge::DT_BF16,
        OP_LOGE(tilingContext->GetNodeName(),
                                        "input var dtype only support float32, float16, bfloat16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->alphaDtype != this->varDtype,
        OP_LOGE(tilingContext->GetNodeName(),
                                        "input alpha and var dtype not same, alpha dtype is %d", this->alphaDtype),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->deltaDtype != this->varDtype,
        OP_LOGE(tilingContext->GetNodeName(),
                                        "input delta and var dtype not same, delta dtype is %d", this->deltaDtype),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyGradientDescentTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->varDtype,
        OP_LOGE(tilingContext->GetNodeName(),
                                        "input and output dtype not same, output dtype is %d", this->deltaDtype),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

ge::graphStatus ApplyGradientDescentTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTiling CheckShape enter.");
    auto varStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, varStorageShape);
    const gert::Shape &varShape = EnsureNotScalar(varStorageShape->GetStorageShape());

    auto deltaStorageShape = tilingContext->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, deltaStorageShape);
    const gert::Shape &deltaShape = EnsureNotScalar(deltaStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape &outputShape = EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(varShape != deltaShape || varShape != outputShape,
        OP_LOGE(tilingContext->GetNodeName(), "input var, delta and output shape not same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyGradientDescentTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ApplyGradientDescentTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    tiling = tilingContext->GetTilingData<ApplyGradientDescentNs::ApplyGradientDescentTilingData>();
    OP_CHECK_IF((tiling == nullptr), OP_LOGE(tilingContext, "Get EleBaseTilingData from context failed"), return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ApplyGradientDescentDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<ApplyGradientDescentDAG<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "output dtype not support");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext, "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4ApplyGradientDescent(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4ApplyGradientDescent rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ApplyGradientDescentCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    ApplyGradientDescentTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyGradientDescent(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ApplyGradientDescentCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyGradientDescent).Tiling(Tiling4ApplyGradientDescent).TilingParse<ApplyGradientDescentCompileInfo>(TilingPrepareForApplyGradientDescent);
} // namespace optiling