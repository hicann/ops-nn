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
 * \file p_relu_tiling_arch35.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "activation/p_relu/op_kernel/arch35/p_relu_struct.h"
#include "activation/p_relu/op_kernel/arch35/p_relu_dag.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "error_util.h"
#include "p_relu_tiling_arch35.h"

using namespace AscendC;
using namespace ge;
using namespace PreluOp;
using namespace Ops::NN::OpTiling;

namespace optiling {
static constexpr uint64_t PRELU_COMMON_TILING_PRIORITY = 0;
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
static const int64_t DIM_NUM_NHWC = 4;
static const int64_t DIM_NUM_NC1HWC0 = 5;
static const int64_t NC1HWC0_DIM_INDEX_C0 = 4;
static const int64_t DIM_NUM_NDC1HWC0 = 6;
static const int64_t NDC1HWC0_DIM_INDEX_C0 = 5;
static const int64_t NDC1HWC0_DIM_INDEX_C1 = 2;

ge::graphStatus PreluTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool PreluTiling::IsCapable()
{
    return true;
}

ge::graphStatus PreluTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "PreluTiling RunTiling enter.");
    auto xInputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputDesc);
    ge::DataType xInputDtype = xInputDesc->GetDataType();
    OP_CHECK_IF(
        xInputDtype != ge::DT_FLOAT16 && xInputDtype != ge::DT_BF16 && xInputDtype != ge::DT_FLOAT,
        OP_LOGE(
            context_->GetNodeName(), "input x dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(xInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto weightInputDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightInputDesc);
    ge::DataType weightInputDtype = weightInputDesc->GetDataType();
    OP_CHECK_IF(
        xInputDtype != weightInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "input weight dtype %s not equal input x dtype %s.",
            ge::TypeUtils::DataTypeToSerialString(weightInputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(xInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        outputDtype != xInputDtype,
        OP_LOGE(
            context_->GetNodeName(), "output y dtype %s not same as input x %s.",
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(xInputDtype).c_str()),
        return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (xInputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<PreluDAG<half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "BroadcastBaseTiling<PreluDAG<half>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xInputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<PreluDAG<bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OPS_REPORT_VECTOR_INNER_ERR(
                context_->GetNodeName(), "BroadcastBaseTiling<PreluDAG<bfloat16_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xInputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<PreluDAG<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "BroadcastBaseTiling<PreluDAG<float>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(), "input dtype %s not supported, only support [float16, bfloat16, float32].",
            ge::TypeUtils::DataTypeToSerialString(xInputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PreluTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t PreluTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus PreluTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PreluTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PreluTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4PRelu(gert::TilingContext* context)
{
    OP_LOGD("PreluTiling", "Enter TilingForPrelu");
    if (context == nullptr) {
        OP_LOGE("PreluTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    PreluTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4PRelu(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingPrepare4PRelu.");
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    OP_LOGD("BroadCastTiling", "Current is regbase soc version.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_LOGD(context->GetNodeName(), "end to do TilingPrepare4PRelu.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PRelu).Tiling(Tiling4PRelu).TilingParse<BroadcastCompileInfo>(TilingPrepare4PRelu);
REGISTER_OPS_TILING_TEMPLATE(PRelu, PreluTiling, PRELU_COMMON_TILING_PRIORITY);
} // namespace optiling