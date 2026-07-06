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
 * \file adaptive_avg_pool2d_grad_simt_tiling.cpp
 * \brief
 */
#include "adaptive_avg_pool2d_grad_simt_tiling.h"

namespace optiling {
static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
static constexpr int64_t MAX_THREAD_NUM = 1024;
bool AdaptiveAvgPool2dGradTilingSimt::IsCapable() { return true; }

ge::graphStatus AdaptiveAvgPool2dGradTilingSimt::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter AdaptiveAvgPool2dGradTilingSimt DoOpTiling.");
    tilingData_->nDim = inputData_.nX;
    tilingData_->cDim = inputData_.cX;
    tilingData_->hInDim = inputData_.hX;
    tilingData_->wInDim = inputData_.wX;
    tilingData_->hOutDim = inputData_.hGrad;
    tilingData_->wOutDim = inputData_.wGrad;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSimt::PostTiling()
{
    int64_t outDataCount = inputData_.nX * inputData_.cX * inputData_.hX * inputData_.wX;
    int64_t threads = std::min(outDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum_));
    context_->SetBlockDim(blockNum);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool2dGradTilingSimt::NeedInt64(int64_t isize, int64_t osize) const
{
    return static_cast<int64_t>(isize) * static_cast<int64_t>(osize) > INT32_MAX;
}
uint64_t AdaptiveAvgPool2dGradTilingSimt::GetTilingKey() const
{
    int64_t outDataCount = inputData_.nX * inputData_.cX * inputData_.hX * inputData_.wX;
    bool needInt64 = (outDataCount > static_cast<int64_t>(INT32_MAX) || NeedInt64(inputData_.hX, inputData_.hGrad) ||
                      NeedInt64(inputData_.wX, inputData_.wGrad) ||
                      inputData_.hGrad > static_cast<int64_t>(INT32_MAX) ||
                      inputData_.wGrad > static_cast<int64_t>(INT32_MAX));
    uint32_t idxDtype = needInt64 ? TPL_INT64 : TPL_INT32;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, idxDtype, isChannelLast);
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradTilingSimt, 50);
} // namespace optiling
