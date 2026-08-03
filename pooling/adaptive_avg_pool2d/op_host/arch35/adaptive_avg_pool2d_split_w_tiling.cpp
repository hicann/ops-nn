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
 * \file adaptive_avg_pool2d_split_w_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool2d_split_w_tiling.h"

constexpr uint64_t SPLIT_W_KERNEL_W_LINE = 32;

namespace optiling {

bool AdaptiveAvgPool2dSplitWTiling::IsCapable()
{
    if (!InitComputeBase(computeInfo_)) {
        return false;
    }

    // SplitW targets downsampling along W with large W kernels that SmallKernel rejects.
    // It loads whole input H-rows and reduces an arbitrarily large W window via the kernel's
    // inner accumulate loop. The H direction is handled by the kernel's hi-reuse loop, so we
    // only gate on W downsampling (wOut<=wIn) regardless of H up/down sampling. This covers
    // cases like W极大下采样 + H上采样 (e.g. kW=316, kW=117).
    bool isWBigKernel = (computeInfo_.kernelWMax >= SPLIT_W_KERNEL_W_LINE);
    bool isWDownsampling = (input_.wOut <= input_.wIn);
    bool isWOutValid = (input_.wOut > 1);
    bool isNcEnough = (input_.nIn * input_.cIn >= computeInfo_.vfLen / TILING_DOUBLE);
    bool isCapable = isWBigKernel && isWDownsampling && isWOutValid && isNcEnough && IsMeetUbSize();

    OP_LOGD(context_->GetNodeName(),
            "AdaptiveAvgPool2dSplitWTiling IsCapable: kHMax=%lu, kWMax=%lu, wIn=%lu, wOut=%lu, hIn=%lu, hOut=%lu, "
            "NC=%lu, isWOutValid=%s, result=%s",
            computeInfo_.kernelHMax, computeInfo_.kernelWMax, input_.wIn, input_.wOut, input_.hIn, input_.hOut,
            input_.nIn * input_.cIn, isWOutValid ? "true" : "false", isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool2dSplitWTiling::CalUbSplitSize() { CalCommonUbSplitSize(computeInfo_, input_.wIn); }

bool AdaptiveAvgPool2dSplitWTiling::IsMeetUbSize()
{
    CalUbSplitSize();
    return CalCommonUbOccupy(computeInfo_, input_.wIn) <= computeInfo_.availableUbSize;
}

ge::graphStatus AdaptiveAvgPool2dSplitWTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dSplitWTiling DoOpTiling start.");

    auto meetUb = [this]() { return IsMeetUbSize(); };
    StandardUbOptimization(computeInfo_, meetUb);

    CalUbSplitSize();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSplitWTiling SetTilingData failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSplitWTiling::SetTilingData()
{
    return FillCommonTilingData<AdaptivePool2dSplitWTilingData>(context_, input_, computeInfo_);
}

void AdaptiveAvgPool2dSplitWTiling::PrintTilingData() const { PrintCommonTilingData(context_, input_, computeInfo_); }

uint64_t AdaptiveAvgPool2dSplitWTiling::GetTilingKey() const
{
    return CalCommonTilingKey(TPL_SPLIT_W_KERNEL, computeInfo_);
}

ge::graphStatus AdaptiveAvgPool2dSplitWTiling::PostTiling()
{
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dSplitWTiling, 2);
} // namespace optiling
