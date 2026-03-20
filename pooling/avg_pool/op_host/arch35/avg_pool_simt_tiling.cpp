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
 * \file avg_pool_simt_tiling.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "pooling/avg_pool_v2/op_host/arch35/avg_pool_v2_common_tiling.h"
#include "avg_pool_simt_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
static constexpr uint64_t POOL_TILING_KEY_SIMT_NCHW_INT32 = 911100;
static constexpr uint64_t POOL_TILING_KEY_SIMT_NHWC_INT32 = 911101;
static constexpr uint64_t POOL_TILING_KEY_SIMT_NCHW_INT64 = 911110;
static constexpr uint64_t POOL_TILING_KEY_SIMT_NHWC_INT64 = 911111;

bool PoolSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus PoolSimtTiling::DoOpTiling()
{
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PoolSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t PoolSimtTiling::GetTilingKey() const
{
    int64_t xSize = inputData.batches * inputData.channels * inputData.inputShape[H_DIM] * inputData.inputShape[W_DIM];
    int64_t ySize = inputData.batches * inputData.channels * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        if (xSize <= INT32_MAX && ySize <= INT32_MAX) {
            OP_LOGI("AvgPoolV2_TilingKey=911100");
            return POOL_TILING_KEY_SIMT_NCHW_INT32;
        }
        OP_LOGI("AvgPoolV2_TilingKey=911110");
        return POOL_TILING_KEY_SIMT_NCHW_INT64;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        if (xSize <= INT32_MAX) {
            OP_LOGI("AvgPoolV2_TilingKey=911101");
            return POOL_TILING_KEY_SIMT_NHWC_INT32;
        }
        OP_LOGI("AvgPoolV2_TilingKey=911111");
        return POOL_TILING_KEY_SIMT_NHWC_INT64;
    }
    return 0;
}

ge::graphStatus PoolSimtTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PoolSimtTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

void PoolSimtTiling::SetTilingData()
{
    AvgPool::AvgPoolSimtTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolSimtTilingData>();
    tilingData->nDim = inputData.batches;
    tilingData->cDim = inputData.channels;
    tilingData->hInDim = inputData.inputShape[H_DIM];
    tilingData->wInDim = inputData.inputShape[W_DIM];
    tilingData->hOutDim = inputData.outShape[H_DIM];
    tilingData->wOutDim = inputData.outShape[W_DIM];
    tilingData->kH = inputData.kernelSize[H_DIM];
    tilingData->kW = inputData.kernelSize[W_DIM];
    tilingData->sH = inputData.stride[H_DIM];
    tilingData->sW = inputData.stride[W_DIM];
    tilingData->tPad = inputData.pad[TOP_PAD_INDEX];
    tilingData->bPad = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->lPad = inputData.pad[LEFT_PAD_INDEX];
    tilingData->rPad = inputData.pad[RIGHT_PAD_INDEX];
    tilingData->divisorOverride = inputData.divisorOverride;
    tilingData->countIncludePad = inputData.countIncludePad ? 1 : 0;
}

void PoolSimtTiling::DumpTilingInfo()
{
    AvgPool::AvgPoolSimtTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolSimtTilingData>();
    std::ostringstream str;
    str << " nDim:" << tilingData->nDim;
    str << " cDim:" << tilingData->cDim;
    str << " hInDim:" << tilingData->hInDim;
    str << " wInDim:" << tilingData->wInDim;
    str << " hOutDim:" << tilingData->hOutDim;
    str << " wOutDim:" << tilingData->wOutDim;
    str << " kH:" << tilingData->kH;
    str << " kW:" << tilingData->kW;
    str << " sH:" << tilingData->sH;
    str << " sW:" << tilingData->sW;
    str << " tPad:" << tilingData->tPad;
    str << " bPad:" << tilingData->bPad;
    str << " lPad:" << tilingData->lPad;
    str << " rPad:" << tilingData->rPad;
    str << " divisorOverride:" << tilingData->divisorOverride;
    str << " countIncludePad:" << tilingData->countIncludePad;
    OP_LOGI("SIMTTILING", "AvgPoolV2 tilingInfo is :%s", str.str().c_str());
}

//////////////////////////////// AvgPoolSimtTiling /////////////////////////////////
ge::graphStatus AvgPoolSimtTiling::GetPlatformInfo()
{
    return GetAvgPoolPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolSimtTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolShapeAttrsInfo(context_, inputData);
}
//////////////////////////////// AvgPoolV2SimtTiling ////////////////////////////////
ge::graphStatus AvgPoolV2SimtTiling::GetPlatformInfo()
{
    return GetAvgPoolV2PlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolV2SimtTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolV2ShapeAttrsInfo(context_, inputData);
}

REGISTER_TILING_TEMPLATE("AvgPool", AvgPoolSimtTiling, 19);
REGISTER_TILING_TEMPLATE("AvgPoolV2", AvgPoolV2SimtTiling, 19);
}  // namespace optiling