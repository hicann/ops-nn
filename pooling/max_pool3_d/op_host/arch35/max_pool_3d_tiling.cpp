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
 * \file max_pool_3d_tiling.cpp
 * \brief
 */

#include "pooling/pool_3d_common/op_host/arch35/pool_tiling_templates_registry.h"
#include "pooling/pool_3d_common/op_host/arch35/max_pool_3d_tiling_common.h"
#include "log/log.h"

using namespace AscendC;
using optiling::PoolTilingRegistry;
namespace optiling {
ge::graphStatus Tiling4MaxPool3D(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling for max_pool_3d is running.");
    auto compileInfoPtr = context->GetCompileInfo<MaxPool3DCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    return PoolTilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4MaxPool3D(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("MaxPool3D", "Context is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MaxPool3DCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3D).Tiling(Tiling4MaxPool3D).TilingParse<MaxPool3DCompileInfo>(TilingPrepare4MaxPool3D);

} // namespace optiling
