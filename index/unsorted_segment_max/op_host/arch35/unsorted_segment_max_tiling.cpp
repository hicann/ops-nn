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
 * \file unsorted_segment_max_tiling.cpp
 * \brief unsorted_segment_max_tiling
 */

#include "unsorted_segment_max_tiling.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace AscendC;

namespace optiling{

static ge::graphStatus Tiling4UnsortedSegmentMax(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling for UnsortedSegmentMax is running");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4UnsortedSegmentMax(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare for UnsortedSegmentMax enter.");

    auto compileInfo = context->GetCompiledInfo<UnsortedSegmentCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->maxThread = Ops::Base::GetSimtMaxThreadNum(context);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm); // 接口预留8K
    compileInfo->ubSizePlatForm = ubSizePlatForm;

    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get corenum size"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "corenum_platform is %lu", compileInfo->coreNum);

    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu", compileInfo->ubSizePlatForm);

    OP_CHECK_IF((compileInfo->maxThread <= 0), OP_LOGE(context, "Failed to get max_thread"),return ge::GRAPH_FAILED);
    OP_LOGD(context, "max_thread_platform is %lu", compileInfo->maxThread);
    OP_LOGD(context, "Tiling Prepare for UnsortedSegmentMax end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UnsortedSegmentMax)
    .Tiling(Tiling4UnsortedSegmentMax)
    .TilingParse<UnsortedSegmentCompileInfo>(TilingPrepare4UnsortedSegmentMax);

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxOutFlTiling, 1);
REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxSimdSplitColTiling, 20);
REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxSimdNonSortTiling, 30);
REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxSimdDynSortTiling, 50);
REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxSortSimtTiling, 60);
REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentMax, UnsortedSegmentMaxSimtTiling, 100);

} //namespace optiling