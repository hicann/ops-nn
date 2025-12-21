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
 * \file adaptive_max_pool3d_grad_tiling.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#include <iostream>

using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

static ge::graphStatus Tiling4AdaptiveMaxPool3DGrad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4AdaptiveMaxPool3DGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4AdaptiveMaxPool3DGrad.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<Tiling4AdaptiveMaxPool3DGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    OP_CHECK_IF(
        (compileInfoPtr->totalCoreNum <= 0), OP_LOGE(context, "Failed to get corenum size"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (compileInfoPtr->maxUbSize <= 0), OP_LOGE(context, "Failed to get maxub size"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdaptiveMaxPool3DGrad)
    .Tiling(Tiling4AdaptiveMaxPool3DGrad)
    .TilingParse<Tiling4AdaptiveMaxPool3DGradCompileInfo>(TilingPrepare4AdaptiveMaxPool3DGrad);
} // namespace optiling
