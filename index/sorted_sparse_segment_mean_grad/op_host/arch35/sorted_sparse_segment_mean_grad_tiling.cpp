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
 * \file sorted_sparse_segment_mean_grad_tiling.cpp
 * \brief
 */

#include "sorted_sparse_segment_mean_grad_tiling_base.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace AscendC;
namespace optiling {
constexpr int32_t DEPENDENCY_INPUT_INDEX = 4;

ge::graphStatus Tiling4SortedSparseSegmentMeanGrad(gert::TilingContext* context) 
{
     return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4SortedSparseSegmentMeanGrad(gert::TilingParseContext* context) 
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<SortedSparseSegmentMeanGradCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SortedSparseSegmentMeanGrad)
    .Tiling(Tiling4SortedSparseSegmentMeanGrad)
    .TilingParse<SortedSparseSegmentMeanGradCompileInfo>(TilingPrepare4SortedSparseSegmentMeanGrad)
    .TilingInputsDataDependency({DEPENDENCY_INPUT_INDEX});

}  // namespace optiling
 