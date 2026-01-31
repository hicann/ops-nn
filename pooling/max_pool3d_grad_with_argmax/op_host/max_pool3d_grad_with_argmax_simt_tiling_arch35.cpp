/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file max_pool3d_grad_with_argmax_simt_tiling_arch35.cpp
 * \brief
 */
#include <iostream>
#include "max_pool3d_grad_with_argmax_tiling_arch35.h"

using namespace MaxPool3DGradWithArgmaxOp;

namespace optiling {

static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
static constexpr int64_t MAX_THREAD_NUM = 1024;

bool MaxPool3DGradWithArgmaxTilingSimt::IsCapable()
{
    return true;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingSimt::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradWithArgmaxTilingSimt DoOpTiling.");
    tilingData_->nDim = inputData.nX;
    tilingData_->cDim = inputData.cX;
    tilingData_->dInDim = inputData.dX;
    tilingData_->hInDim = inputData.hX;
    tilingData_->wInDim = inputData.wX;
    tilingData_->dOutDim = inputData.dGrad;
    tilingData_->hOutDim = inputData.hGrad;
    tilingData_->wOutDim = inputData.wGrad;
    tilingData_->kSizeD = inputData.dKernel;
    tilingData_->kSizeH = inputData.hKernel;
    tilingData_->kSizeW = inputData.wKernel;
    tilingData_->strideD = inputData.dStride;
    tilingData_->strideH = inputData.hStride;
    tilingData_->strideW = inputData.wStride;
    tilingData_->padD = inputData.dPad;
    tilingData_->padH = inputData.hPad;
    tilingData_->padW = inputData.wPad;
    tilingData_->dilationD = inputData.dDilation;
    tilingData_->dilationH = inputData.hDilation;
    tilingData_->dilationW = inputData.wDilation;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingSimt::PostTiling()
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    int64_t threads = std::min(outDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum_));
    context_->SetBlockDim(blockNum);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DGradWithArgmaxTilingSimt::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = (inputData.inputFormat == ge::Format::FORMAT_NDHWC) ? 1 : 0;
    uint32_t isSimt = 1;
    uint32_t isCheckRange = 0;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, isCheckRange);
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxTilingSimt, 5);
}  // namespace optiling
