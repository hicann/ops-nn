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
 * \file max_pool_grad_with_argmax_v3_tiling.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "max_pool_grad_with_argmax_v3_tiling_base.h"

using namespace AscendC;
using Ops::NN::Optiling::TilingRegistry;
namespace optiling {

ge::graphStatus Tiling4MaxPoolGradWithArgmaxV3(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4MaxPoolGradWithArgmaxV3(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<MaxPoolGradWithArgmaxCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolGradWithArgmaxV3)
    .Tiling(Tiling4MaxPoolGradWithArgmaxV3)
    .TilingParse<MaxPoolGradWithArgmaxCompileInfo>(TilingPrepare4MaxPoolGradWithArgmaxV3);

} // namespace optiling
