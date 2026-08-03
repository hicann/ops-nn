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
 * \file adaptive_avg_pool2d_split_c_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool2d_split_c_tiling.h"

constexpr uint64_t SPLIT_C_KERNEL_W_LINE = 32;
constexpr uint64_t SPLIT_C_KERNEL_W_MAX = 1000;
constexpr uint64_t SPLIT_C_SMALL_KERNEL_LIMIT = 128;
constexpr uint64_t SPLIT_C_MIN_NC_FOR_LOW_HIN = 64;
constexpr uint64_t SPLIT_C_LOW_HIN_THRESHOLD = 4;
constexpr uint64_t SPLIT_C_KLARGE_MIN_HOUT = 10;

namespace optiling {

bool AdaptiveAvgPool2dSplitCTiling::IsCapable()
{
    if (!InitComputeBase(computeInfo_)) {
        return false;
    }
    computeInfo_.wInFactor = 1; // minimum wi-chunk for IsCapable; DoOpTiling maximizes

    // SplitC is the W-downsampling fallback (priority 5) tried after SplitW (priority 3).
    // NC must be at least vfLen/2 for vectorization to have sufficient efficiency;
    // below that threshold, the vector utilization is too low and BigKernel/SIMT
    // handles the case more efficiently. This matches SplitW's NC threshold.
    // For H-upsampling (hOut > hIn), the kernel uses WReduceChunkToSum + ScatterSumToHo to
    // W-reduce each (hi, chunk) once and scatter to all covering ho positions.
    bool isWBigKernel = (computeInfo_.kernelWMax >= SPLIT_C_KERNEL_W_LINE);
    bool isKernelLarge = (computeInfo_.kernelHMax * computeInfo_.kernelWMax >= SPLIT_C_SMALL_KERNEL_LIMIT);
    bool isHUpsampling = (input_.hOut > input_.hIn);
    bool isWKernelBounded = (computeInfo_.kernelWMax < SPLIT_C_KERNEL_W_MAX);
    bool isWDownsampling = (input_.wOut <= input_.wIn);
    bool isNcEnough = (input_.nIn * input_.cIn >= computeInfo_.vfLen / TILING_DOUBLE);
    bool isHOutValid = (input_.hOut > 1);
    uint64_t nc = input_.nIn * input_.cIn;
    bool isShapeEfficient = !(input_.hIn <= SPLIT_C_LOW_HIN_THRESHOLD && nc < SPLIT_C_MIN_NC_FOR_LOW_HIN);
    bool isKLargeHOutTooSmall = (isKernelLarge && !isWBigKernel && !isHUpsampling &&
                                 input_.hOut <= SPLIT_C_KLARGE_MIN_HOUT);
    bool isCapable = (isWBigKernel || isKernelLarge || isHUpsampling) && !isKLargeHOutTooSmall && isWKernelBounded &&
                     isWDownsampling && isNcEnough && isHOutValid && isShapeEfficient && IsMeetUbSize();

    OP_LOGD(context_->GetNodeName(),
            "AdaptiveAvgPool2dSplitCTiling IsCapable: kHMax=%lu, kWMax=%lu, wIn=%lu, wOut=%lu, hIn=%lu, hOut=%lu, "
            "NC=%lu, isHUpsampling=%s, isHOutValid=%s, isShapeEfficient=%s, isWKernelBounded=%s, "
            "isKLargeHOutTooSmall=%s, result=%s",
            computeInfo_.kernelHMax, computeInfo_.kernelWMax, input_.wIn, input_.wOut, input_.hIn, input_.hOut, nc,
            isHUpsampling ? "true" : "false", isHOutValid ? "true" : "false", isShapeEfficient ? "true" : "false",
            isWKernelBounded ? "true" : "false", isKLargeHOutTooSmall ? "true" : "false", isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool2dSplitCTiling::CalUbSplitSize() { CalCommonUbSplitSize(computeInfo_, computeInfo_.wInFactor); }

bool AdaptiveAvgPool2dSplitCTiling::IsMeetUbSize()
{
    CalUbSplitSize();
    return CalCommonUbOccupy(computeInfo_, computeInfo_.wInFactor) <= computeInfo_.availableUbSize;
}

ge::graphStatus AdaptiveAvgPool2dSplitCTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dSplitCTiling DoOpTiling start.");

    auto meetUb = [this]() { return IsMeetUbSize(); };
    computeInfo_.ncFactor = computeInfo_.vfLen;
    computeInfo_.hiFactor = 1;
    computeInfo_.hoFactor = 1;

    BinarySearchMaxWInFactor(computeInfo_, input_.wIn, meetUb);

    if (input_.hOut <= input_.hIn && computeInfo_.kernelHMax > 1) {
        ShrinkHiFactor(computeInfo_, meetUb);
    }
    BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);

    if (computeInfo_.xDtypeSize == TILING_DOUBLE &&
        (computeInfo_.hoFactor <= 1 || computeInfo_.kernelHMax * computeInfo_.kernelWMax > TILING_LARGE_KERNEL_AREA)) {
        TryHalfNcFactorSplitC(computeInfo_, meetUb);
    }

    OptimizeHoFactorForCores(computeInfo_);

    if (input_.hOut > input_.hIn) {
        OptimizeHoForHUpsample(computeInfo_, meetUb);
    }

    CalUbSplitSize();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSplitCTiling SetTilingData failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSplitCTiling::SetTilingData()
{
    auto ret = FillCommonTilingData<AdaptivePool2dSplitCTilingData>(context_, input_, computeInfo_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    auto* tilingData = context_->GetTilingData<AdaptivePool2dSplitCTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->wInFactor = static_cast<int64_t>(computeInfo_.wInFactor);
    return ge::GRAPH_SUCCESS;
}

void AdaptiveAvgPool2dSplitCTiling::PrintTilingData() const
{
    PrintCommonTilingData(context_, input_, computeInfo_);
    OP_LOGI(context_->GetNodeName(), "wInFactor: %lu", computeInfo_.wInFactor);
}

uint64_t AdaptiveAvgPool2dSplitCTiling::GetTilingKey() const
{
    return CalCommonTilingKey(TPL_SPLIT_C_KERNEL, computeInfo_);
}

ge::graphStatus AdaptiveAvgPool2dSplitCTiling::PostTiling()
{
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dSplitCTiling, 4);
} // namespace optiling
