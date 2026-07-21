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
 * \file threshold_grad_v2_d_tiling_arch35.cpp
 * \brief threshold_grad_v2_d_tiling_arch35
 */

#include "threshold_grad_v2_d_tiling.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/threshold_grad_v2_d_struct.h"
#include "../../op_kernel/arch35/threshold_grad_v2_d_dag.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace ThresholdGradV2DOp;

namespace optiling {
static constexpr uint64_t THRESHOLD_GRAD_V2_D_COMMON_TILING_PRIORITY = 0;

ge::graphStatus ThresholdGradV2DTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

bool ThresholdGradV2DTiling::IsCapable() { return true; }

ge::graphStatus ThresholdGradV2DTiling::DoOpTiling()
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
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "gradOutput, self, out",
                                               ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                                                   ge::TypeUtils::DataTypeToSerialString(input1DType) + ", " +
                                                   ge::TypeUtils::DataTypeToSerialString(outputDtype),
                                               "The dtypes of gradOutput, self and out must be the same");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float thresHold = scaleValueAttr != nullptr ? *scaleValueAttr : 1.0f;
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;

    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<ThresholdGradV2DDag<half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<half>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_FP16);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<ThresholdGradV2DDag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<bfloat16_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_BF16);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<ThresholdGradV2DDag<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<float>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_FP32);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<ThresholdGradV2DDag<int32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<int32_t>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_INT32);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<ThresholdGradV2DDag<int8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<int8_t>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_INT8);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<ThresholdGradV2DDag<uint8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2DDag<uint8_t>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_D_TPL_UINT8);
        brcBaseTiling.SetScalar<float>(thresHold);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gradOutput, self",
                                  ge::TypeUtils::DataTypeToSerialString(input0DType),
                                  "DT_FLOAT16, DT_BF16, DT_FLOAT,DT_INT32, DT_INT8, DT_UINT8");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThresholdGradV2DTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ThresholdGradV2DTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus ThresholdGradV2DTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ThresholdGradV2DTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ThresholdGradV2DTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus TilingForThresholdGradV2D(gert::TilingContext* context)
{
    OP_LOGD("ThresholdGradV2DTiling", "Enter TilingForThresholdGradV2D");
    if (context == nullptr) {
        OP_LOGE("ThresholdGradV2DTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("ThresholdGradV2DTiling", "Enter new ThresholdGradV2DTiling");
    ThresholdGradV2DTiling tiling(context);
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

IMPL_OP_OPTILING(ThresholdGradV2D)
    .Tiling(TilingForThresholdGradV2D)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(ThresholdGradV2D, ThresholdGradV2DTiling, THRESHOLD_GRAD_V2_D_COMMON_TILING_PRIORITY);
} // namespace optiling
