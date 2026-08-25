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
static constexpr int64_t MIN_THREAD_NUM = 512;
static constexpr int64_t SMALL_SPATIAL_THRESHOLD = 512;
static constexpr int64_t TINY_SPATIAL_THRESHOLD = 4;
static constexpr int64_t DEEP_REDUCE_THRESHOLD = 64;
const int64_t MAX_INT32 = 2147483647;
// SIMT kernel 的 DIV_T 在 int32 档为 uint32_t，magic 除法对 dividend 的合法范围
// 是整个 uint32，只有 divisor（各轴长）受框架 uint32 magic 公式限制必须 <= INT32_MAX。
const uint64_t MAX_UINT32 = 4294967295ULL;

bool AdaptiveAvgPool3dGradTilingSimt::IsCapable() { return true; }

int64_t AdaptiveAvgPool3dGradTilingSimt::GetSimtThreadNum() const
{
    const int64_t outSpatial = inputData.dX * inputData.hX * inputData.wX;
    // 每个输出元素累加的梯度点数：各轴上采样比的乘积，不上采样的轴记 1。
    const int64_t reduceD = (inputData.dGrad > inputData.dX) ? Ops::Base::CeilDiv(inputData.dGrad, inputData.dX) : 1;
    const int64_t reduceH = (inputData.hGrad > inputData.hX) ? Ops::Base::CeilDiv(inputData.hGrad, inputData.hX) : 1;
    const int64_t reduceW = (inputData.wGrad > inputData.wX) ? Ops::Base::CeilDiv(inputData.wGrad, inputData.wX) : 1;
    const int64_t reduceVolume = reduceD * reduceH * reduceW;

    if (outSpatial < TINY_SPATIAL_THRESHOLD && reduceVolume >= DEEP_REDUCE_THRESHOLD) {
        return MIN_THREAD_NUM;
    }
    return (outSpatial < SMALL_SPATIAL_THRESHOLD) ? MAX_THREAD_NUM : MIN_THREAD_NUM;
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
    tilingData_->threadMode = (GetSimtThreadNum() == MIN_THREAD_NUM) ? TPL_THREAD_512 : TPL_THREAD_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSimt::PostTiling()
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    int64_t threads = std::min(outDataCount, GetSimtThreadNum());
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
    int64_t inDataCount = inputData.nGrad * inputData.cGrad * inputData.dGrad * inputData.hGrad * inputData.wGrad;
    // kernel 以 n*c*inD*inH*inW 作为 uint32 循环上界，输入元素总数也必须 <= UINT32_MAX，
    // outDataCount <= MAX_INT32 保证输出侧偏移不超界；
    bool needInt64 = (outDataCount > static_cast<int64_t>(MAX_INT32) ||
                      inDataCount > static_cast<int64_t>(MAX_UINT32) || NeedInt64(inputData.dX, inputData.dGrad) ||
                      NeedInt64(inputData.hX, inputData.hGrad) || NeedInt64(inputData.wX, inputData.wGrad));
    uint32_t idxDtype = needInt64 ? TPL_INT64 : TPL_INT32;
    uint32_t isChannelLast = (inputData.inputFormat == ge::Format::FORMAT_NDHWC) ? 1 : 0;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, idxDtype, isChannelLast);
}

REGISTER_TILING_TEMPLATE("AdaptiveAvgPool3dGrad", AdaptiveAvgPool3dGradTilingSimt, 50);
} // namespace optiling
