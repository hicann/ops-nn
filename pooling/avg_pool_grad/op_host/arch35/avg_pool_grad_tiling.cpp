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
 * \file avg_pool_grad_tiling.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "avg_pool_grad_tiling_base.h"
#include "error_util.h"

namespace optiling
{
using Ops::NN::Optiling::TilingRegistry;
ge::graphStatus Tiling4AvgPoolGrad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4AvgPoolGrad(gert::TilingParseContext* context) {
   OP_LOGD("AvgPoolGrad", "TilingPrepare4AvgPoolGrad");
   fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
   OP_TILING_CHECK(platformInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "platformInfoPtr info is null"),
       return ge::GRAPH_FAILED);
   auto compileInfoPtr = context->GetCompiledInfo<AvgPoolGradCompileInfo>();
   OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compileInfoPtr is null"),
       return ge::GRAPH_FAILED);
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
   compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
   return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AvgPoolGrad)
    .InputsDataDependency({0})
    .Tiling(Tiling4AvgPoolGrad)
    .TilingParse<AvgPoolGradCompileInfo>(TilingPrepare4AvgPoolGrad);
}  // namespace optiling