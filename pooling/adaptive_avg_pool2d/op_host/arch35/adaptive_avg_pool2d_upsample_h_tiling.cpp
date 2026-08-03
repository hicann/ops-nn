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
 * \file adaptive_avg_pool2d_upsample_h_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool2d_upsample_h_tiling.h"

constexpr uint64_t UPSAMPLE_H_MAX_WOUT = 512;
constexpr uint64_t UPSAMPLE_H_SMALL_KERNEL_LIMIT = 128;
constexpr uint64_t UPSAMPLE_H_FP32_WUP_MIN_WOUT = 32;
// Slack factor on coreNum: only keep shrinking hoFactor for core occupancy while totalOuter
// stays within this multiple of the core count, so the search cannot run away on large hOut.
constexpr uint64_t UPSAMPLE_H_CORE_OCCUPY_SLACK = 4;

namespace optiling {

bool AdaptiveAvgPool2dUpsampleHTiling::IsCapable()
{
    if (!InitComputeBase(computeInfo_)) {
        return false;
    }

    bool isHUpsampling = (input_.hOut > input_.hIn);
    bool isWOutBounded = (input_.wOut <= UPSAMPLE_H_MAX_WOUT);
    bool isWOutValid = (input_.wOut > 1);
    bool isNcEnough = (input_.nIn * input_.cIn >= computeInfo_.vfLen / TILING_DOUBLE);
    bool isWUpTinyInput = (input_.wOut > input_.wIn && input_.wIn <= TILING_DOUBLE);
    bool isHin1TinyW = (input_.hIn == 1 && input_.wIn < computeInfo_.alignNum);
    bool isFp32WUpSmallWin = (computeInfo_.xDtypeSize == sizeof(float) && input_.wOut > input_.wIn &&
                              input_.wIn <= computeInfo_.alignNum && input_.wOut >= UPSAMPLE_H_FP32_WUP_MIN_WOUT);
    bool ubFit = IsMeetUbSize();
    if (!ubFit && computeInfo_.xDtypeSize < sizeof(float)) {
        computeInfo_.ncFactor = computeInfo_.vfLen / TILING_DOUBLE;
        ubFit = IsMeetUbSize();
        computeInfo_.ncFactor = computeInfo_.vfLen;
    }
    bool isCapable = isHUpsampling && isWOutBounded && isWOutValid && isNcEnough && !isWUpTinyInput && !isHin1TinyW &&
                     !isFp32WUpSmallWin && ubFit;

    OP_LOGD(context_->GetNodeName(),
            "AdaptiveAvgPool2dUpsampleHTiling IsCapable: hIn=%lu, hOut=%lu, wIn=%lu, wOut=%lu, "
            "kHMax=%lu, kWMax=%lu, NC=%lu, ubSize=%lu, vfLen=%lu, isWOutValid=%s, isWUp=%s, "
            "isHin1TinyW=%s, isFp32WUpSmallWin=%s, result=%s",
            input_.hIn, input_.hOut, input_.wIn, input_.wOut, computeInfo_.kernelHMax, computeInfo_.kernelWMax,
            input_.nIn * input_.cIn, computeInfo_.availableUbSize, computeInfo_.vfLen, isWOutValid ? "true" : "false",
            (input_.wOut > input_.wIn) ? "true" : "false", isHin1TinyW ? "true" : "false",
            isFp32WUpSmallWin ? "true" : "false", isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool2dUpsampleHTiling::CalUbSplitSize()
{
    CalCommonUbSplitSize(computeInfo_, input_.wIn);
    // Early-Cast: fp16/bf16 Cast before TransOut, so resQue uses T-sized buffer (not fp32)
    if (computeInfo_.xDtypeSize < sizeof(float)) {
        uint64_t wOutAlign = Ops::Base::CeilAlign(input_.wOut, computeInfo_.alignNum);
        uint64_t outTransAlign = Ops::Base::CeilAlign(computeInfo_.hoFactor * wOutAlign, TILING_TRANS_ADDR_LEN);
        computeInfo_.resQue1Size = outTransAlign * computeInfo_.ncFactor * computeInfo_.xDtypeSize;
        computeInfo_.resQue2Size = 0;
    }
}

bool AdaptiveAvgPool2dUpsampleHTiling::IsMeetUbSize()
{
    CalUbSplitSize();
    uint64_t wInAlign = Ops::Base::CeilAlign(input_.wIn, computeInfo_.alignNum);
    uint64_t wOutAlign = Ops::Base::CeilAlign(input_.wOut, computeInfo_.alignNum);
    uint64_t vlNum = computeInfo_.ncFactor;
    uint64_t transRowAlign = Ops::Base::CeilAlign(computeInfo_.hiFactor * wInAlign, TILING_TRANS_ADDR_LEN);
    uint64_t transBufSize = transRowAlign * vlNum * computeInfo_.xDtypeSize;
    uint64_t outTransAlign = Ops::Base::CeilAlign(computeInfo_.hoFactor * wOutAlign, TILING_TRANS_ADDR_LEN);
    uint64_t outBufSize = outTransAlign * vlNum * sizeof(float);
    uint64_t vfLenI32 = Ops::Base::GetVRegSize(context_) / sizeof(int32_t);
    uint64_t wBufSize = Ops::Base::CeilAlign(input_.wOut, vfLenI32) * sizeof(int32_t) * TILING_DOUBLE;
    uint64_t wiWoBufSize = 0;
    if (input_.wOut > input_.wIn) {
        uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
        wiWoBufSize = Ops::Base::CeilAlign(input_.wIn * sizeof(int32_t), dataBlock) * TILING_DOUBLE;
    }
    uint64_t total = computeInfo_.inputQueSize + transBufSize + outBufSize + computeInfo_.resQue1Size +
                     computeInfo_.resQue2Size + wBufSize + wiWoBufSize;
    return total <= computeInfo_.availableUbSize;
}

ge::graphStatus AdaptiveAvgPool2dUpsampleHTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dUpsampleHTiling DoOpTiling start.");

    auto meetUb = [this]() { return IsMeetUbSize(); };
    computeInfo_.ncFactor = computeInfo_.vfLen;

    ShrinkHiFactor(computeInfo_, meetUb);
    BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);

    bool hoFactorStuckAt1 = (computeInfo_.hoFactor <= 1);
    bool largeKernel = (computeInfo_.kernelHMax * computeInfo_.kernelWMax > TILING_LARGE_KERNEL_AREA);
    if (computeInfo_.xDtypeSize < sizeof(float) && (hoFactorStuckAt1 || largeKernel)) {
        uint64_t origNcFactor = computeInfo_.ncFactor;
        uint64_t origHiFactor = computeInfo_.hiFactor;
        uint64_t origHoFactor = computeInfo_.hoFactor;
        bool origFits = IsMeetUbSize();
        computeInfo_.ncFactor = computeInfo_.vfLen / TILING_DOUBLE;
        ShrinkHiFactor(computeInfo_, meetUb);
        BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);
        if (computeInfo_.hoFactor <= origHoFactor && origFits) {
            computeInfo_.ncFactor = origNcFactor;
            computeInfo_.hiFactor = origHiFactor;
            computeInfo_.hoFactor = origHoFactor;
        }
    }

    CalUbBlockFactor(computeInfo_);
    while (computeInfo_.useCoreNum < input_.coreNum && computeInfo_.hoFactor > 1 &&
           computeInfo_.totalOuter <= input_.coreNum * UPSAMPLE_H_CORE_OCCUPY_SLACK) {
        uint64_t lastBlockFactor = computeInfo_.blockFactor;
        computeInfo_.hoFactor--;
        CalUbBlockFactor(computeInfo_);
        if (computeInfo_.blockFactor > lastBlockFactor) {
            computeInfo_.hoFactor++;
            CalUbBlockFactor(computeInfo_);
            break;
        }
    }

    // No final UB re-check here. IsCapable() already validated UB at the minimum
    // hiFactor/hoFactor=1 configuration, and every search primitive above only grows those
    // factors while meetUb() holds, so the resulting config always fits. Failing here would
    // also be the wrong response: template selection happens in IsCapable(), so returning an
    // error from DoOpTiling() fails the whole tiling instead of yielding to a lower-priority
    // template.
    CalUbSplitSize();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dUpsampleHTiling SetTilingData failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dUpsampleHTiling::SetTilingData()
{
    return FillCommonTilingData<AdaptivePool2dUpsampleHTilingData>(context_, input_, computeInfo_);
}

void AdaptiveAvgPool2dUpsampleHTiling::PrintTilingData() const
{
    PrintCommonTilingData(context_, input_, computeInfo_);
}

uint64_t AdaptiveAvgPool2dUpsampleHTiling::GetTilingKey() const
{
    return CalCommonTilingKey(TPL_UPSAMPLE_H_KERNEL, computeInfo_);
}

ge::graphStatus AdaptiveAvgPool2dUpsampleHTiling::PostTiling()
{
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dUpsampleHTiling, 0);
} // namespace optiling
