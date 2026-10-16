/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !\
 * \file rms_norm_hi_f4_quant_tiling.cpp
 * \brief
 */
#include "rms_norm_hi_f4_quant_tiling.h"

namespace optiling {
static ge::graphStatus Tiling4RmsNormHiF4Quant(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context, "RmsNormHiF4Quant is not supported on the current chip!");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4RmsNormHiF4Quant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("RmsNormHiF4Quant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4RmsNormHiF4Quant.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<RmsNormHiF4QuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfoPtr->totalCoreNum <= 0),
        OP_LOGE(context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfoPtr->totalCoreNum)),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->totalUbSize);
    OP_CHECK_IF(
        (compileInfoPtr->totalUbSize <= 0),
        OP_LOGE(context, "Get block Size failed, block size: %u", static_cast<uint32_t>(compileInfoPtr->totalUbSize)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormHiF4Quant)
    .Tiling(Tiling4RmsNormHiF4Quant)
    .TilingParse<RmsNormHiF4QuantCompileInfo>(TilingPrepare4RmsNormHiF4Quant);
} // namespace optiling
