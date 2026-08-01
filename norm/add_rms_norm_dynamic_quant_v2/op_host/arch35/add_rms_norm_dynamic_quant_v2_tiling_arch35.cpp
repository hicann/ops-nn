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
 * \file add_rms_norm_dynamic_quant_v2_tiling_arch35.cpp
 * \brief
 */

#include "op_host/tiling_util.h"
#include "op_api/runtime2_util.h"
#include "add_rms_norm_dynamic_quant_v2_tiling_arch35.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr int GAMMA_IDX = 2;

static ge::graphStatus TilingPrepare4AddRmsNormDynamicQuantV2(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuantv2", "Context is null"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4AddRmsNormDynamicQuantV2.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "PlatformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<AddRmsNormDynamicQuantV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "CompileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AddRmsNormDynamicQuantV2(gert::TilingContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuantV2", "Context is null"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Tiling4AddRmsNormDynamicQuantV2");
    auto colShape = context->GetInputShape(GAMMA_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, colShape);
    auto colStorageShape = optiling::EnsureNotScalar(colShape->GetStorageShape());
    uint32_t colVal = colStorageShape.GetDim(0);
    if (colVal == 0) {
        AddRmsNormDynamicQuantEmptyTiling emptyTiling(context);
        return emptyTiling.DoTiling();
    }
    AddRmsNormDynamicQuantRegbaseTiling regbaseTiling(context);
    return regbaseTiling.DoTiling();
}

// register tiling interface of AddRmsNormDynamicQuantV2 op.
IMPL_OP_OPTILING(AddRmsNormDynamicQuantV2)
    .Tiling(Tiling4AddRmsNormDynamicQuantV2)
    .TilingParse<AddRmsNormDynamicQuantV2CompileInfo>(TilingPrepare4AddRmsNormDynamicQuantV2);
} // namespace optiling
