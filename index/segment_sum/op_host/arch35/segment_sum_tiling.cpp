/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file segment_sum_tiling.cpp
 * \brief segment_sum_tiling
 */

#include "segment_sum_tiling_base.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace AscendC;
namespace optiling {

ge::graphStatus TilingSegmentSumForAscendC(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus Tiling4SegmentSum(gert::TilingContext* context) 
{
    OP_LOGD(context->GetNodeName(), "SegmentSumTiling is running.");
    auto compile_info = reinterpret_cast<const SegmentSumCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return TilingSegmentSumForAscendC(context);
}

ge::graphStatus TilingPrepare4SegmentSum(gert::TilingParseContext* context) 
{
    OP_LOGD(context->GetNodeName(), "Start to do TilingPrepare for SegmentSum.");
    
    auto compileInfoPtr = context->GetCompiledInfo<SegmentSumCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ub_size);
    
    OP_CHECK_IF((compileInfoPtr->core_num <= 0),
                    OP_LOGE(context->GetNodeName(),
                    "The core num is invaild."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF((compileInfoPtr->ub_size <= 0),
                    OP_LOGE(context->GetNodeName(),
                    "The UB size from platform is invaild."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4SegmentSum is end.");
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SegmentSum)
    .Tiling(Tiling4SegmentSum)
    .TilingParse<SegmentSumCompileInfo>(TilingPrepare4SegmentSum);

} // namespace optiling