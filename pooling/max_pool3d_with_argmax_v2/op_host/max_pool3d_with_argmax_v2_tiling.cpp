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
 * \file max_pool3d_with_argmax_v2_tiling.cpp
 * \brief
 */

#include "max_pool3d_with_argmax_v2_tiling_base.h"
using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

static ge::graphStatus Tiling4MaxPool3DWithArgmaxV2(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MaxPool3DWithArgmaxV2(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MaxPool3DWithArgmaxV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);

    OP_CHECK_IF(
        compileInfoPtr->coreNum <= 0,
        OP_LOGE(
            context->GetNodeName(), "MaxPool3DWithArgmaxV2: coreNum = %zu, should be greater than 0",
            compileInfoPtr->coreNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compileInfoPtr->ubSize <= 0,
        OP_LOGE(
            context->GetNodeName(), "MaxPool3DWithArgmaxV2: ubSize = %zu, should be greater than 0",
            compileInfoPtr->ubSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3DWithArgmaxV2)
    .Tiling(Tiling4MaxPool3DWithArgmaxV2)
    .TilingParse<MaxPool3DWithArgmaxV2CompileInfo>(TilingPrepare4MaxPool3DWithArgmaxV2);

} // namespace optiling
