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
 * \file max_pool_with_argmax_v3_simt_tiling.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "platform/platform_ascendc.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "avg_pool_v2_grad_simt_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace AvgPoolV2Grad;

namespace optiling {

ge::graphStatus AvgPoolV2GradTilingSIMT::DoOpTiling()
{
    OP_LOGD(context_, "Begin to do AvgPoolV2GradTilingSIMT::DoOpTiling");
    AvgPoolV2GradSimtTilingData* tilingData = context_->GetTilingData<AvgPoolV2GradSimtTilingData>();
    tilingData->nDim = inputData.batches;
    tilingData->cDim = inputData.channels;
    tilingData->hInDim = inputData.outShape[H_DIM];
    tilingData->wInDim = inputData.outShape[W_DIM];
    tilingData->hPooledDim = inputData.gradShape[H_DIM];
    tilingData->wPooledDim = inputData.gradShape[W_DIM];
    tilingData->kSizeH = inputData.kernelSize[H_DIM];
    tilingData->kSizeW = inputData.kernelSize[W_DIM];
    tilingData->stridesH = inputData.stride[H_DIM];
    tilingData->stridesW = inputData.stride[W_DIM];
    tilingData->padHLeft = inputData.pad[PAD_HL_DIM];
    tilingData->padHRight = inputData.pad[PAD_HR_DIM];
    tilingData->padWLeft = inputData.pad[PAD_WL_DIM];
    tilingData->padWRight = inputData.pad[PAD_WR_DIM];
    tilingData->countIncludePad = inputData.countIncludePad;
    tilingData->divisorOverride = inputData.divisorOverride;

    int64_t outputDataCount = tilingData->nDim * tilingData->cDim * tilingData->hInDim * tilingData->wInDim;
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum));
    context_->SetBlockDim(blockNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t AvgPoolV2GradTilingSIMT::GetTilingKey() const
{
    uint64_t schMode = TPL_SIMT_KERNEL;
    uint64_t format = TPL_NCHW_FORMAT;
    if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        format = TPL_NHWC_FORMAT;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        schMode, format, static_cast<uint64_t>(inputData.isInt32Meet), TPL_NO_PAD, TPL_NO_CHECK_RANGE,
        inputData.countIncludePad, static_cast<uint64_t>(inputData.hasDivisor));

    return tilingKey;
}

ge::graphStatus AvgPoolV2GradTilingSIMT::GetPlatformInfo() {
    return GetAvgPoolV2GradPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolV2GradTilingSIMT::GetShapeAttrsInfo() {
    return GetAvgPoolV2GradShapeAttrsInfo(context_, inputData);
}

bool AvgPoolV2GradTilingSIMT::IsCapable()
{
    return true;
}

ge::graphStatus AvgPoolV2GradTilingSIMT::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}
     
ge::graphStatus AvgPoolV2GradTilingSIMT::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}
     
ge::graphStatus AvgPoolV2GradTilingSIMT::PostTiling()
{
    ubSize = ubSize - DCACHE_SIZE;
    context_->SetLocalMemorySize(ubSize);

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AvgPoolV2Grad, AvgPoolV2GradTilingSIMT, 100);
} // namespace optiling