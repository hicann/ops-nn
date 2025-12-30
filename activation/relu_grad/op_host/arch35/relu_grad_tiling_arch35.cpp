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
 * \file relu_grad_tiling_arch35.cpp
 * \brief relu_grad_tiling
 */

#include "relu_grad_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling_base/tiling_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "activation/relu_grad/op_kernel/arch35/relu_grad_struct.h"
#include "activation/relu_grad/op_kernel/arch35/relu_grad_dag.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;
using namespace ReluGradOp;

namespace optiling
{
static constexpr uint64_t RELU_GRAD_COMMON_TILING_PRIORITY = 0;

ge::graphStatus ReluGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool ReluGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus ReluGradTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (inputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<ReluGradDag<half>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_FP16);
    } else if (inputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<ReluGradDag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_BF16);
    } else if (inputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<ReluGradDag<float>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_FP32);
    } else if (inputDtype == ge::DT_INT8) {
        BroadcastBaseTiling<ReluGradCastDag<int8_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_INT8);
    } else if (inputDtype == ge::DT_UINT8) {
        BroadcastBaseTiling<ReluGradCastDag<uint8_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_UINT8);
    } else if (inputDtype == ge::DT_INT32) {
        BroadcastBaseTiling<ReluGradDag<int32_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_INT32);
    } else if (inputDtype == ge::DT_INT64) {
        BroadcastBaseTiling<ReluGradDag<int64_t>::OpDag> brcBaseTiling(context_);
        status = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), RELU_GRAD_TPL_INT32);
    } else {
        OP_LOGE(
            context_->GetNodeName(), "Input dtype only support fp16, bf16, fp32, int8, uint8, int32, int64, currently is %s.", 
            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        status != ge::GRAPH_SUCCESS, OP_LOGE(context_, "BroadcastBaseTiling do tiling failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ReluGradTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus ReluGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReluGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForReluGrad(gert::TilingContext* context)
{
    OP_LOGD("ReluGradTiling", "Enter TilingForReluGrad");
    if (context == nullptr) {
        OP_LOGE("ReluGradTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("ReluGradTiling", "Enter new ReluGradTiling");
    ReluGradTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReluGrad).Tiling(TilingForReluGrad).TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(ReluGrad, ReluGradTiling, RELU_GRAD_COMMON_TILING_PRIORITY);
}  // namespace optiling
