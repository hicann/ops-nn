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
 * \file leaky_relu_grad_tiling.cpp
 * \brief leaky_relu_grad_tiling
 */

#include "leaky_relu_grad_tiling.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/leaky_relu_grad_struct.h"
#include "../../op_kernel/arch35/leaky_relu_grad_dag.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace LeakyReluGradOp;

namespace optiling
{
static constexpr uint64_t LEAKY_RELU_GRAD_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LeakyReluGradTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LeakyReluGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus LeakyReluGradTiling::DoOpTiling()
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
    if ((input0DType != input1DType) || (outputDtype != input1DType)) {
    OP_LOGE(context_->GetNodeName(), "dtype of gradients[%s], dtype of features[%s], dtype of backprops[%s] are not equal.",
        Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str(),
        Ops::Base::ToString(static_cast<ge::DataType>(input1DType)).c_str(),
        Ops::Base::ToString(static_cast<ge::DataType>(outputDtype)).c_str());
    return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float negativeSlope = scaleValueAttr != nullptr ? *scaleValueAttr : 0.0;
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LeakyReluGradDag<half>::OpDag> brcBaseTiling(context_);
        baseTilingResult=brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<LeakyReluGradDag<half>::OpDag> failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LEAKY_RELU_GRAD_TPL_FP16);
        brcBaseTiling.SetScalar<float>(negativeSlope);
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<LeakyReluGradDag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult=brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<LeakyReluGradDag<bfloat16_t>::OpDag> failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LEAKY_RELU_GRAD_TPL_BF16);
        brcBaseTiling.SetScalar<float>(negativeSlope);
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LeakyReluGradDag<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult=brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<LeakyReluGradDag<float>::OpDag> failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), LEAKY_RELU_GRAD_TPL_FP32);
        brcBaseTiling.SetScalar<float>(negativeSlope);
    } else {
        OP_LOGE(context_->GetNodeName(), "Input0[gradients]:%s and input1[features]:%s is only support float16, bfloat16, float32",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str(), Ops::Base::ToString(static_cast<ge::DataType>(input1DType)).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LeakyReluGradTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LeakyReluGradTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluGradTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LeakyReluGradTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLeakyReluGrad(gert::TilingContext* context)
{
    OP_LOGD("LeakyReluGradTiling", "Enter TilingForLeakyReluGrad");
    if (context == nullptr) {
        OP_LOGE("LeakyReluGradTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("LeakyReluGradTiling", "Enter new LeakyReluGradTiling");
    LeakyReluGradTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext *context)
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

IMPL_OP_OPTILING(LeakyReluGrad).Tiling(TilingForLeakyReluGrad).TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(LeakyReluGrad, LeakyReluGradTiling, LEAKY_RELU_GRAD_COMMON_TILING_PRIORITY);
}  // namespace optiling