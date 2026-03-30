/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_avg_pool3d_grad_simt_tiling.cpp
 * \brief
 */
#include <iostream>
#include "adaptive_avg_pool3d_grad_tiling_arch35.h"

using namespace AdaptiveAvgPool3dGradOp;
namespace optiling {

static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
static constexpr int64_t MAX_THREAD_NUM = 1024;
const int64_t MAX_INT32 = 2147483647;

bool AdaptiveAvgPool3dGradTilingSimt::IsCapable()
{
    return true; 
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSimt::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveAvgPool3dGradTilingSimt DoOpTiling.");
    tilingData_->nDim = inputData.nX;
    tilingData_->cDim = inputData.cX;
    tilingData_->dInDim = inputData.dX;
    tilingData_->hInDim = inputData.hX;
    tilingData_->wInDim = inputData.wX;
    tilingData_->dOutDim = inputData.dGrad;
    tilingData_->hOutDim = inputData.hGrad;
    tilingData_->wOutDim = inputData.wGrad;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSimt::PostTiling()
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    int64_t threads = std::min(outDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum_));
    context_->SetBlockDim(blockNum);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool3dGradTilingSimt::NeedInt64(int64_t isize, int64_t osize) const
{
    return static_cast<int64_t>(isize) * static_cast<int64_t>(osize) > MAX_INT32;
}
uint64_t AdaptiveAvgPool3dGradTilingSimt::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    bool needInt64 = (outDataCount > static_cast<int64_t>(MAX_INT32) || 
                      NeedInt64(inputData.dX, inputData.dGrad) ||
                      NeedInt64(inputData.hX, inputData.hGrad) ||
                      NeedInt64(inputData.wX, inputData.wGrad));
    uint32_t idxDtype = needInt64 ? TPL_INT64 : TPL_INT32;
    uint32_t isChannelLast = (inputData.inputFormat == ge::Format::FORMAT_NDHWC) ? 1 : 0;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, idxDtype, isChannelLast);
}

REGISTER_TILING_TEMPLATE("AdaptiveAvgPool3dGrad", AdaptiveAvgPool3dGradTilingSimt, 50);
}  // namespace optiling
