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
 * \file adaptive_avg_pool2d_tiling.cpp
 * \brief
 */
#include <iostream>

#include "adaptive_avg_pool2d_tiling.h"
using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

static ge::graphStatus Tiling4AdaptiveAvgPool2d(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4AdaptiveAvgPool2d(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4AdaptiveAvgPool2d.");
    auto compileInfo = context->GetCompiledInfo<AdaptiveAvgPool2dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4AdaptiveAvgPool2d end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveAvgPool2d)
    .Tiling(Tiling4AdaptiveAvgPool2d)
    .TilingParse<AdaptiveAvgPool2dCompileInfo>(TilingPrepare4AdaptiveAvgPool2d);
} // namespace optiling
