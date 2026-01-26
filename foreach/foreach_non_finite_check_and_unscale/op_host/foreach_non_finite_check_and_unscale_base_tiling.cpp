/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_non_finite_check_and_unscale_base_tiling.cpp
 * \brief
 */
#include "foreach_non_finite_check_and_unscale_tiling.h"

namespace optiling {
ge::graphStatus ForeachNonFiniteCheckAndUnscaleBaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    npuArch = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleBaseClass::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleBaseClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleBaseClass::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ForeachNonFiniteCheckAndUnscale(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4ForeachNonFiniteCheckAndUnscale([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachNonFiniteCheckAndUnscale)
    .Tiling(Tiling4ForeachNonFiniteCheckAndUnscale)
    .TilingParse<ForeachNonFiniteCheckAndUnscaleCompileInfo>(TilingPrepare4ForeachNonFiniteCheckAndUnscale);
} // namespace optiling
