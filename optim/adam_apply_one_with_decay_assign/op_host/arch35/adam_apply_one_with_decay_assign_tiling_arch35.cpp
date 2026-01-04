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
 * \file adam_apply_one_with_decay_assign_tiling_arch35.cpp
 * \brief adam_apply_one_with_decay_assign_tiling_arch35 source file
 */

#include "adam_apply_one_with_decay_assign_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/adam_apply_one_with_decay_assign_dag.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_COMMON_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_NUM = 11;
constexpr static int32_t OUTPUT_NUM = 3;
constexpr static int32_t INPUT4_INDEX = 4;
constexpr static int32_t BETA1_INDEX = 5;
constexpr static int32_t BETA2_INDEX = 6;
constexpr static int32_t BETA3_INDEX = 7;
constexpr static int32_t BETA4_INDEX = 8;
constexpr static int32_t BETA5_INDEX = 9;
constexpr static int32_t ADD2_INDEX = 10;

static ge::graphStatus TilingPrepareForAdamApplyOneWithDecayAssign(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AdamApplyOneWithDecayAssignCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::GetShapeAttrsInfo()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    for (int32_t inputIdx = 1; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = context_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);

        auto curDtype = inputDesc->GetDataType();
        if (curDtype != input0DType) {
            OP_CHECK_IF(
                curDtype != input0DType,
                OP_LOGE(context_->GetNodeName(), "Input %d dtype not match with input0 dtype.", inputIdx),
                return ge::GRAPH_FAILED);
        }
    }
    for (int32_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        auto outputDesc = context_->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

        auto curDtype = outputDesc->GetDataType();
        if (curDtype != input0DType) {
            OP_CHECK_IF(
                curDtype != input0DType,
                OP_LOGE(context_->GetNodeName(), "Output %d dtype not match with input0 dtype.", outputIdx),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool AdamApplyOneWithDecayAssignTiling::IsCapable()
{
    return true;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::DoOpTiling()
{
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Shape check failed."),
        return ge::GRAPH_FAILED);

    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<AdamApplyOneWithDecayAssignOp::AdamApplyOneWithDecayAssignCompute<half, float>::OpDag>
            brcBaseTiling(context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_BOTH));
        OP_CHECK_IF(
            brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<AdamApplyOneWithDecayAssignOp::AdamApplyOneWithDecayAssignCompute<bfloat16_t, float>::OpDag>
            brcBaseTiling(context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_BOTH));
        OP_CHECK_IF(
            brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<AdamApplyOneWithDecayAssignOp::AdamApplyOneWithDecayAssignCompute<float, float>::OpDag>
            brcBaseTiling(context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_BOTH));
        OP_CHECK_IF(
            brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(
            context_->GetNodeName(), "input dtype is only support fp16, bf16, fp32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool AdamApplyOneWithDecayAssignTiling::CheckIsScalar(int32_t inputIdx)
{
    auto inputShape = context_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return true;
    }
    return false;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::CheckShape()
{
    OP_CHECK_IF(
        !CheckIsScalar(INPUT4_INDEX), OP_LOGE(context_->GetNodeName(), "Input η must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(BETA1_INDEX), OP_LOGE(context_->GetNodeName(), "Input β1 must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(BETA2_INDEX), OP_LOGE(context_->GetNodeName(), "Input β2 must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(BETA3_INDEX), OP_LOGE(context_->GetNodeName(), "Input β3 must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(BETA4_INDEX), OP_LOGE(context_->GetNodeName(), "Input β4 must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(BETA5_INDEX), OP_LOGE(context_->GetNodeName(), "Input λ must be scalar."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckIsScalar(ADD2_INDEX), OP_LOGE(context_->GetNodeName(), "Input ϵ must be scalar."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AdamApplyOneWithDecayAssignTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneWithDecayAssignTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForAdamApplyOneWithDecayAssign(gert::TilingContext* context)
{
    OP_LOGD("AdamApplyOneWithDecayAssignTiling", "Enter TilingForAdamApplyOneWithDecayAssign");
    if (context == nullptr) {
        OP_LOGE("AdamApplyOneWithDecayAssignTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AdamApplyOneWithDecayAssignCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc AdamApplyOneTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(AdamApplyOneWithDecayAssign)
    .Tiling(TilingForAdamApplyOneWithDecayAssign)
    .TilingParse<AdamApplyOneWithDecayAssignCompileInfo>(TilingPrepareForAdamApplyOneWithDecayAssign);

REGISTER_OPS_TILING_TEMPLATE(
    AdamApplyOneWithDecayAssign, AdamApplyOneWithDecayAssignTiling,
    ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_COMMON_TILING_PRIORITY);
} // namespace optiling