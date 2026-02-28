/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_tiling.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "avg_pool_v2_grad_tiling_base.h"
#include "error_util.h"

namespace optiling
{
using Ops::NN::Optiling::TilingRegistry;
ge::graphStatus Tiling4AvgPoolV2Grad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4AvgPoolV2Grad(gert::TilingParseContext* context) {
   OP_LOGD("AvgPoolGrad", "TilingPrepare4AvgPoolGrad");
   fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
   OP_TILING_CHECK(platformInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "platformInfoPtr info is null"),
       return ge::GRAPH_FAILED);
   auto compileInfoPtr = context->GetCompiledInfo<AvgPoolV2GradCompileInfo>();
   OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compileInfoPtr is null"),
       return ge::GRAPH_FAILED);
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
   compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
   return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AvgPoolV2Grad)
    .InputsDataDependency({0})
    .Tiling(Tiling4AvgPoolV2Grad)
    .TilingParse<AvgPoolV2GradCompileInfo>(TilingPrepare4AvgPoolV2Grad);
}  // namespace optiling