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
 * \file adam_apply_one_tiling_arch35.cpp
 * \brief adam_apply_one_tiling_arch35 source file
 */

#include "adam_apply_one_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "../../op_kernel/arch35/adam_apply_one_dag.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t ADAM_APPLY_ONE_COMMON_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_NUM = 10;
constexpr static int32_t OUTPUT_NUM = 3;

static ge::graphStatus TilingPrepareForAdamApplyOne(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AdamApplyOneCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneTiling::GetShapeAttrsInfo()
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
        auto outputDesc = context_->GetInputDesc(outputIdx);
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

bool AdamApplyOneTiling::IsCapable()
{
    return true;
}

ge::graphStatus AdamApplyOneTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<AdamApplyOneOp::AdamApplyOneCompute<half, half>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(
            brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<AdamApplyOneOp::AdamApplyOneCompute<bfloat16_t, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(
            brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<AdamApplyOneOp::AdamApplyOneCompute<float, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
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

ge::graphStatus AdamApplyOneTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AdamApplyOneTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AdamApplyOneTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdamApplyOneTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForAdamApplyOne(gert::TilingContext* context)
{
    OP_LOGD("AdamApplyOneTiling", "Enter TilingForAdamApplyOne");
    if (context == nullptr) {
        OP_LOGE("AdamApplyOneTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AdamApplyOneCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc AdamApplyOneTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(AdamApplyOne)
    .Tiling(TilingForAdamApplyOne)
    .TilingParse<AdamApplyOneCompileInfo>(TilingPrepareForAdamApplyOne);

REGISTER_OPS_TILING_TEMPLATE(AdamApplyOne, AdamApplyOneTiling, ADAM_APPLY_ONE_COMMON_TILING_PRIORITY);
} // namespace optiling