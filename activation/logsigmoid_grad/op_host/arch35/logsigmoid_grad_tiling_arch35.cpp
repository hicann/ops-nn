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
 * \file logsigmoid_grad_tiling_arch35.cc
 * \brief logsigmoid_grad_tiling
 */

#include "logsigmoid_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "activation/logsigmoid_grad/op_kernel/arch35/logsigmoid_grad_dag.h"
#include "activation/logsigmoid_grad/op_kernel/arch35/logsigmoid_grad_struct.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace LogSigmoidGradOp;

namespace optiling {
static constexpr uint64_t LOGSIGMOID_GRAD_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LogSigmoidGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LogSigmoidGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus LogSigmoidGradTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if ((input0DType != input1DType) || (outputDtype != input0DType)) {
        OP_LOGE(
            context_->GetNodeName(),
            "dtype of gradOutput[%s], dtype of self[%s], dtype of gradInput[%s] are not equal.",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str(),
            Ops::Base::ToString(static_cast<ge::DataType>(input1DType)).c_str(),
            Ops::Base::ToString(static_cast<ge::DataType>(outputDtype)).c_str());
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus status = ge::GRAPH_FAILED;
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LogSigmoidGradDag<half>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<LogSigmoidGradDag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LogSigmoidGradDag<float>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(),
            "Input0[gradOutput]:%s and input1[self]:%s is only support float16, bfloat16, float32",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str(),
            Ops::Base::ToString(static_cast<ge::DataType>(input1DType)).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling do tiling failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogSigmoidGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LogSigmoidGradTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LogSigmoidGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogSigmoidGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogSigmoidGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLogSigmoidGrad(gert::TilingContext* context)
{
    OP_LOGD("LogSigmoidGrad", "Enter TilingForLogSigmoidGrad");
    if (context == nullptr) {
        OP_LOGE("LogSigmoidGrad", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("LogSigmoidGradTiling", "Enter ascendc LogSigmoidGradTiling");
    LogSigmoidGradTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogSigmoidGrad)
    .Tiling(TilingForLogSigmoidGrad)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(LogSigmoidGrad, LogSigmoidGradTiling, LOGSIGMOID_GRAD_COMMON_TILING_PRIORITY);
} // namespace optiling
