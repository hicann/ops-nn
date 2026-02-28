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
 * \file gelu_grad_tiling_arch35.cpp
 * \brief
 */
#include "gelu_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "activation/gelu_grad/op_kernel/arch35/gelu_grad_dag.h"
#include "activation/gelu_grad/op_kernel/arch35/gelu_grad_struct.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace GeluGradOp;

namespace optiling {
static constexpr uint64_t GELU_GRAD_COMMON_TILING_PRIORITY = 0;
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const int64_t INPUT_Y_INDEX = 2;

ge::graphStatus GeluGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool GeluGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus GeluGradTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "GeluGradTiling RunTiling enter.");
    auto dyInputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyInputDesc);
    ge::DataType dyInputDtype = dyInputDesc->GetDataType();
    OP_CHECK_IF(
        dyInputDtype != ge::DT_FLOAT16 && dyInputDtype != ge::DT_BF16 && dyInputDtype != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(), "input dy dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto xInputDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    ge::DataType xInputDtype = xInputDesc->GetDataType();
    OP_CHECK_IF(
        xInputDtype != dyInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "input x dtype %s not equal dy dtype %s.",
            ge::TypeUtils::DataTypeToSerialString(xInputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto yInputDesc = context_->GetInputDesc(INPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yInputDesc);
    ge::DataType yInputDtype = yInputDesc->GetDataType();
    OP_CHECK_IF(
        yInputDtype != dyInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "input y dtype %s not equal dy dtype %s.",
            ge::TypeUtils::DataTypeToSerialString(yInputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        outputDtype != dyInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "output z dtype %s not same as input dy %s.",
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (dyInputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<GeluGradDAG<half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<GeluGradDag<half>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (dyInputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<GeluGradDAG<bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<GeluGradDag<bfloat16_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (dyInputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<GeluGradDAG<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<GeluGradDag<float>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(), "input dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GeluGradTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus GeluGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GeluGrad(gert::TilingContext* tilingContextGen)
{
    OP_CHECK_IF(
        tilingContextGen == nullptr, OP_LOGE("Tiling4GeluGrad", "Tiling context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContextGen->GetNodeName(), "Enter Tiling4GeluGrad");
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4GeluGrad rt2.0 is running.");
    GeluGradTiling tiling(tilingContextGen);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForGeluGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GeluGrad).Tiling(Tiling4GeluGrad).TilingParse<ElewiseCompileInfo>(TilingPrepareForGeluGrad);

REGISTER_OPS_TILING_TEMPLATE(GeluGrad, GeluGradTiling, GELU_GRAD_COMMON_TILING_PRIORITY);
} // namespace optiling