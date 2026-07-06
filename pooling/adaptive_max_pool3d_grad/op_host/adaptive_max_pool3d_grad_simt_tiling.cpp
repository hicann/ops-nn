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
 * \file adaptive_max_pool3d_grad_simt_tiling.cpp
 * \brief
 */
#include <iostream>
#include "adaptive_max_pool3d_grad_tiling_arch35.h"

using namespace AdaptiveMaxPool3dGradOp;
namespace optiling {

static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;

bool AdaptiveMaxPool3dGradTilingSimt::IsCapable() { return true; }

ge::graphStatus AdaptiveMaxPool3dGradTilingSimt::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveMaxPool3dGradTilingSimt DoOpTiling.");
    tilingData_->nDim = inputData.nX;
    tilingData_->cDim = inputData.cX;
    tilingData_->dInDim = inputData.dGrad;
    tilingData_->hInDim = inputData.hGrad;
    tilingData_->wInDim = inputData.wGrad;
    tilingData_->dOutDim = inputData.dX;
    tilingData_->hOutDim = inputData.hX;
    tilingData_->wOutDim = inputData.wX;
    tilingData_->isOverLap = (inputData.dX % inputData.dGrad != 0 || inputData.hX % inputData.hGrad != 0 ||
                              inputData.wX % inputData.wGrad != 0) ?
                                 1 :
                                 0;
    tilingData_->deterministicFlag = (context_->GetDeterministic() == 1 && tilingData_->isOverLap) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingSimt::PostTiling()
{
    int64_t blockNum;
    if (tilingData_->deterministicFlag) {
        int64_t totalNC = inputData.nX * inputData.cX;
        constexpr int64_t detThreadDim = 512;
        blockNum = Ops::Base::CeilDiv(totalNC, detThreadDim);
    } else {
        int64_t outDataCount = inputData.nX * inputData.cX * inputData.dGrad * inputData.hGrad * inputData.wGrad;
        blockNum = (outDataCount == 0) ? 1 : std::min(outDataCount, static_cast<int64_t>(coreNum_));
    }
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum_));
    blockNum = std::max(blockNum, static_cast<int64_t>(1));
    context_->SetBlockDim(blockNum);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dGradTilingSimt::GetWorkspaceSize()
{
    constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(WS_SYS_SIZE);
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveMaxPool3dGradTilingSimt::GetTilingKey() const
{
    uint32_t idxDtype = (inputData.argmaxDtype == ge::DT_INT64) ? TPL_INT64 : TPL_INT32;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, idxDtype, 0);
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveMaxPool3DGrad, AdaptiveMaxPool3dGradTilingSimt, 1);
} // namespace optiling
