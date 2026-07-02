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
 * \file scatter_nd_max_tiling.cpp
 * \brief scatter_nd_max_tiling
 */

#include "scatter_nd_max_tiling.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace AscendC;

namespace optiling{

static ge::graphStatus Tiling4ScatterNdMax(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling for ScatterNdMax is running");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4ScatterNdMax(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare for ScatterNdMax enter.");

    auto compileInfo = context->GetCompiledInfo<ScatterNdCommonCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm); // 接口预留8K
    compileInfo->ub_size = ubSizePlatForm;

    OP_CHECK_IF((compileInfo->core_num <= 0), OP_LOGE(context, "Failed to get core_num size"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "corenum_platform is %lu", compileInfo->core_num);

    OP_CHECK_IF((compileInfo->ub_size <= 0), OP_LOGE(context, "Failed to get ub size"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu", compileInfo->ub_size);

    OP_LOGD(context, "Tiling Prepare for ScatterNdMax end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterNdMax)
    .Tiling(Tiling4ScatterNdMax)
    .TilingParse<ScatterNdCommonCompileInfo>(TilingPrepare4ScatterNdMax);

REGISTER_TILING_TEMPLATE("ScatterNdMax", ScatterNdMaxSimdSortTiling, 2);
REGISTER_TILING_TEMPLATE("ScatterNdMax", ScatterNdMaxSimtSortTiling, 5);
REGISTER_TILING_TEMPLATE("ScatterNdMax", ScatterNdMaxSimtTiling, 8);

} //namespace optiling


