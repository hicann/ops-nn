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
 * \file gelu_grad_v2_tiling.cpp
 * \brief gelu_grad_v2_tiling
 */
#include "gelu_grad_v2_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../op_kernel/arch35/gelu_grad_v2_dag.h"
#include "../op_kernel/arch35/gelu_grad_v2_struct.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace GeluGradV2Op;

namespace optiling
{
static constexpr uint64_t GELU_GRAD_V2_COMMON_TILING_PRIORITY = 0;
const int64_t ASCEND_WORKSPACE = 0; // 0M
const int64_t ASCEND_API_BUFFER = 122880; //120K
const int ATTR_APPROXIMATE_POS = 0;
 
ge::graphStatus GeluGradV2Tiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool GeluGradV2Tiling::IsCapable()
{
    return true;
}

ge::graphStatus GeluGradV2Tiling::CheckValid() 
{
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
    
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != dyInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "output z dtype %s not same as input dy %s.",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(dyInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
 	OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto *approximatePtr = attrs->GetAttrPointer<char>(ATTR_APPROXIMATE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, approximatePtr);
    approximateStr = approximatePtr;
    if (approximateStr == "none") {
        approximate = TPL_NONE;
    } else if (approximateStr == "tanh") {
        approximate = TPL_TANH;
    } else {
        OP_LOGE(context_->GetNodeName(), "approximate [%s] not supported, only support [none, tanh]", approximateStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradV2Tiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "GeluGradV2Tiling RunTiling enter.");
    OP_CHECK_IF(CheckValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "validity check failed"), 
               return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (approximate == TPL_NONE) {
        if (this->outputDtype == ge::DT_FLOAT16) {
            BroadcastBaseTiling<GeluGradV2None16DAG<half>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling(ASCEND_API_BUFFER);
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else if (this->outputDtype == ge::DT_BF16) {
            BroadcastBaseTiling<GeluGradV2None16DAG<bfloat16_t>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling(ASCEND_API_BUFFER);
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else if (this->outputDtype == ge::DT_FLOAT) {
            BroadcastBaseTiling<GeluGradV2None32DAG<float>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling(ASCEND_API_BUFFER);
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else {
            OP_LOGE(context_->GetNodeName(), "input dtype %s not supported, only support [float16, bfloat16, float32].", ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<GeluGradV2NoneDAG::OpDag> failed"), return ge::GRAPH_FAILED);
    } else if (approximate == TPL_TANH) {
        if (this->outputDtype == ge::DT_FLOAT16) {
            BroadcastBaseTiling<GeluGradV2TanhDAG<half>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling();
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else if (this->outputDtype == ge::DT_BF16) {
            BroadcastBaseTiling<GeluGradV2TanhDAG<bfloat16_t>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling();
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else if (this->outputDtype == ge::DT_FLOAT) {
            BroadcastBaseTiling<GeluGradV2TanhDAG<float>::OpDag> brcBaseTiling(context_);
            baseTilingResult=brcBaseTiling.DoTiling();
            tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(),approximate);
        } else {
            OP_LOGE(context_->GetNodeName(), "input dtype %s not supported, only support [float16, bfloat16, float32].", ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<GeluGradV2TanhDAG::OpDag> failed"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_->GetNodeName(), "approximate [%s] not supported, only support [none, tanh]", approximateStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GeluGradV2Tiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus GeluGradV2Tiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradV2Tiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluGradV2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GeluGradV2(gert::TilingContext* tilingContextGen)
{
    OP_CHECK_IF(
        tilingContextGen == nullptr, OP_LOGE("Tiling4GeluGradV2", "Tiling context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContextGen->GetNodeName(), "Enter Tiling4GeluGradV2");
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4GeluGradV2 rt2.0 is running.");
    GeluGradV2Tiling tiling(tilingContextGen);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForGeluGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(GeluGradV2).Tiling(Tiling4GeluGradV2).TilingParse<ElewiseCompileInfo>(TilingPrepareForGeluGradV2);

REGISTER_OPS_TILING_TEMPLATE(GeluGradV2, GeluGradV2Tiling, GELU_GRAD_V2_COMMON_TILING_PRIORITY);
} // namespace optiling