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
 * \file instance_norm_tiling.cpp
 * \brief
 */

#include "instance_norm_tiling.h"

namespace optiling {
static ge::graphStatus Tiling4InstanceNorm(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context, "InstanceNorm is not supported on the current chip!");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4InstanceNorm(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4InstanceNorm enter.");

    auto compileInfo = context->GetCompiledInfo<InstanceNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(context, "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
        return ge::GRAPH_FAILED);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        (compileInfo->ubBlockSize <= 0),
        OP_LOGE(
            context, "Get block Size failed, block size: %u",
            static_cast<uint32_t>(compileInfo->ubBlockSize)),
        return ge::GRAPH_FAILED);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        (compileInfo->vectorLength <= 0),
        OP_LOGE(
            context, "Get vector Length failed, vector Length: %u",
            static_cast<uint32_t>(compileInfo->vectorLength)),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4InstanceNorm exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InstanceNorm).Tiling(Tiling4InstanceNorm).TilingParse<InstanceNormCompileInfo>(TilingPrepare4InstanceNorm);

} // namespace optiling
