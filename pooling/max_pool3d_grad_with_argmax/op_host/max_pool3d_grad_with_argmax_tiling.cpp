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
 * \file max_pool3d_grad_with_argmax_tiling.cpp
 * \brief
 */
#include <iostream>

#include "max_pool3d_grad_with_argmax_tiling.h"
using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

static ge::graphStatus Tiling4MaxPool3DGradWithArgmax(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MaxPool3DGradWithArgmax(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4MaxPool3DGradWithArgmax.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<Tiling4MaxPool3DGradWithArgmaxCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3DGradWithArgmax)
    .Tiling(Tiling4MaxPool3DGradWithArgmax)
    .TilingParse<Tiling4MaxPool3DGradWithArgmaxCompileInfo>(TilingPrepare4MaxPool3DGradWithArgmax);
} // namespace optiling
