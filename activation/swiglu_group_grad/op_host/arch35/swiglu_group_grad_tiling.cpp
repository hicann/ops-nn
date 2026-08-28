/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling.cpp
 * \brief SwigluGroupGrad tiling registration (arch35, Ascend950)
 */

#include "op_host/tiling_templates_registry.h"
#include "swiglu_group_grad_tiling_base.h"

namespace optiling {

using Ops::NN::Optiling::TilingRegistry;

ge::graphStatus Tiling4SwigluGroupGradArch35(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4SwigluGroupGradArch35(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluGroupGradTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<SwigluGroupGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluGroupGrad)
    .Tiling(Tiling4SwigluGroupGradArch35)
    .TilingParse<SwigluGroupGradCompileInfo>(TilingPrepare4SwigluGroupGradArch35);

} // namespace optiling
