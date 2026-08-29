/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_simt_tiling.cpp
 * \brief SIMT scheme tiling for 3D average pooling backward (arch35).
 */

#include <algorithm>
#include <string>

#include "op_host/tiling_templates_registry.h"
#include "avg_pool3_d_grad_simt_tiling.h"

namespace optiling {
using namespace AvgPool3DGrad;

static constexpr int64_t MAX_THREAD_NUM = 256;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

bool AvgPool3DGradSimtTiling::IsCapable() { return true; }

uint64_t AvgPool3DGradSimtTiling::GetTilingKey() const
{
    int64_t outCount = inputData.batches * inputData.channels * inputData.gradShape[D_DIM] *
                       inputData.gradShape[H_DIM] * inputData.gradShape[W_DIM];
    uint32_t isInt32Meet = outCount <= static_cast<int64_t>(INT32_MAX) ? TPL_INT32 : TPL_INT64;
    uint32_t format = (inputData.inputFormat == ge::Format::FORMAT_NCDHW) ? TPL_NCDHW_FORMAT : TPL_NDHWC_FORMAT;
    uint32_t countIncludePad = inputData.countIncludePad ? TPL_COUNT_PAD : TPL_NO_COUNT_PAD;
    uint32_t hasDivisor = inputData.hasDivisor;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, format, isInt32Meet, TPL_NO_PAD, TPL_NO_CHECK_RANGE, countIncludePad,
                              hasDivisor);
}

ge::graphStatus AvgPool3DGradSimtTiling::DoOpTiling()
{
    AvgPool3DGradSimtTilingData* tilingData = context_->GetTilingData<AvgPool3DGradSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    AvgPool3DCommon c;
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        c.nDim = 0;
        c.cDim = 1;
        c.dDim = 2;
        c.hDim = 3;
        c.wDim = 4;
    } else {
        c.nDim = 0;
        c.dDim = 1;
        c.hDim = 2;
        c.wDim = 3;
        c.cDim = 4;
    }
    tilingData->nDim = inputData.batches;
    tilingData->cDim = inputData.channels;
    tilingData->dInDim = inputData.inputShape[D_DIM];
    tilingData->hInDim = inputData.inputShape[H_DIM];
    tilingData->wInDim = inputData.inputShape[W_DIM];
    tilingData->dOutDim = inputData.gradShape[D_DIM];
    tilingData->hOutDim = inputData.gradShape[H_DIM];
    tilingData->wOutDim = inputData.gradShape[W_DIM];
    tilingData->kSizeD = inputData.kernelSize[D_DIM];
    tilingData->kSizeH = inputData.kernelSize[H_DIM];
    tilingData->kSizeW = inputData.kernelSize[W_DIM];
    tilingData->stridesD = inputData.stride[D_DIM];
    tilingData->stridesH = inputData.stride[H_DIM];
    tilingData->stridesW = inputData.stride[W_DIM];
    tilingData->padDLeft = inputData.pad[FRONT_PAD_INDEX];
    tilingData->padDRight = inputData.pad[BACKEND_PAD_INDEX];
    tilingData->padHLeft = inputData.pad[TOP_PAD_INDEX];
    tilingData->padHRight = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->padWLeft = inputData.pad[LEFT_PAD_INDEX];
    tilingData->padWRight = inputData.pad[RIGHT_PAD_INDEX];
    tilingData->countIncludePad = inputData.countIncludePad;
    tilingData->divisorOverride = inputData.divisorOverride;

    int64_t outputDataCount = tilingData->nDim * tilingData->cDim * tilingData->dInDim * tilingData->hInDim *
                              tilingData->wInDim;
    int64_t threads = std::min<int64_t>(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum));
    context_->SetBlockDim(blockNum);
    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradSimtTiling::PostTiling()
{
    ubSize = ubSize - DCACHE_SIZE;
    context_->SetLocalMemorySize(ubSize);

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

void AvgPool3DGradSimtTiling::DumpTilingInfo()
{
    AvgPool3DGradSimtTilingData* tilingData = context_->GetTilingData<AvgPool3DGradSimtTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "tilingData is nullptr!");
        return;
    }
    std::string str;
    str += "nDim:" + std::to_string(tilingData->nDim);
    str += ",cDim:" + std::to_string(tilingData->cDim);
    str += ",dInDim:" + std::to_string(tilingData->dInDim);
    str += ",hInDim:" + std::to_string(tilingData->hInDim);
    str += ",wInDim:" + std::to_string(tilingData->wInDim);
    str += ",dOutDim:" + std::to_string(tilingData->dOutDim);
    str += ",hOutDim:" + std::to_string(tilingData->hOutDim);
    str += ",wOutDim:" + std::to_string(tilingData->wOutDim);
    str += ",kSizeD:" + std::to_string(tilingData->kSizeD);
    str += ",kSizeH:" + std::to_string(tilingData->kSizeH);
    str += ",kSizeW:" + std::to_string(tilingData->kSizeW);
    str += ",stridesD:" + std::to_string(tilingData->stridesD);
    str += ",stridesH:" + std::to_string(tilingData->stridesH);
    str += ",stridesW:" + std::to_string(tilingData->stridesW);
    str += ",padDLeft:" + std::to_string(tilingData->padDLeft);
    str += ",padDRight:" + std::to_string(tilingData->padDRight);
    str += ",padHLeft:" + std::to_string(tilingData->padHLeft);
    str += ",padHRight:" + std::to_string(tilingData->padHRight);
    str += ",padWLeft:" + std::to_string(tilingData->padWLeft);
    str += ",padWRight:" + std::to_string(tilingData->padWRight);
    str += ",countIncludePad:" + std::to_string(tilingData->countIncludePad);
    str += ",divisorOverride:" + std::to_string(tilingData->divisorOverride);
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(AvgPool3DGrad, AvgPool3DGradSimtTiling, 6);

} // namespace optiling
