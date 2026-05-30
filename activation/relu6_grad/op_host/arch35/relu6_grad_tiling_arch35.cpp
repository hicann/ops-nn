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
 * \file relu6_grad_tiling_arch35.cpp
 * \brief
 */
#include "relu6_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "activation/relu6_grad/op_kernel/arch35/relu6_grad_dag.h"
#include "activation/relu6_grad/op_kernel/arch35/relu6_grad_struct.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace Relu6GradOp;

namespace optiling {
static constexpr uint64_t RELU6_GRAD_COMMON_TILING_PRIORITY = 0;
static constexpr int64_t FEATURES_INPUT_INDEX = 1;

ge::graphStatus Relu6GradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool Relu6GradTiling::IsCapable()
{
    return true;
}

ge::graphStatus Relu6GradTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Relu6GradTiling RunTiling enter.");
    auto gradInputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradInputDesc);
    ge::DataType gradInputDtype = gradInputDesc->GetDataType();
    OP_CHECK_IF(
        gradInputDtype != ge::DT_FLOAT16 && gradInputDtype != ge::DT_BF16 && gradInputDtype != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(),
            "input gradients dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto featInputDesc = context_->GetInputDesc(FEATURES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, featInputDesc);
    ge::DataType featInputDtype = featInputDesc->GetDataType();
    OP_CHECK_IF(
        featInputDtype != gradInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "input features dtype %s not equal gradients dtype %s.",
            ge::TypeUtils::DataTypeToSerialString(featInputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        outputDtype != gradInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "output backprops dtype %s not same as input gradients %s.",
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (gradInputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<Relu6GradFloatCast<half, float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<Relu6GradFloatCast<half, float>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (gradInputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<Relu6GradFloatCast<bfloat16_t, float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<Relu6GradFloatCast<bfloat16_t, float>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (gradInputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<Relu6Grad<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<Relu6Grad<float>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(),
            "input dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(gradInputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Relu6GradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t Relu6GradTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus Relu6GradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Relu6GradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Relu6GradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Relu6Grad(gert::TilingContext* tilingContextGen)
{
    OP_CHECK_IF(
        tilingContextGen == nullptr, OP_LOGE("Tiling4Relu6Grad", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    OP_LOGD(tilingContextGen->GetNodeName(), "Enter Tiling4Relu6Grad");
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Relu6Grad rt2.0 is running.");
    Relu6GradTiling tiling(tilingContextGen);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForRelu6Grad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Relu6Grad).Tiling(Tiling4Relu6Grad).TilingParse<ElewiseCompileInfo>(TilingPrepareForRelu6Grad);

REGISTER_OPS_TILING_TEMPLATE(Relu6Grad, Relu6GradTiling, RELU6_GRAD_COMMON_TILING_PRIORITY);
} // namespace optiling
