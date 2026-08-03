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
 * \file adaptive_avg_pool2d_split_h_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "adaptive_avg_pool2d_split_h_tiling.h"

// SplitH benefits any genuine H-downsampling (hOut<hIn): vector Add count is the same,
// but outBuf read-modify-write drops O(hIn)->O(hOut). Small kernelH is in fact SplitH's best
// form (one batch per Ho window, isFirst always true, single outBuf write per Ho), so the
// admission gate is hOut<hIn itself rather than any kernelHMax floor.
constexpr uint64_t SPLIT_H_WIN_THRESHOLD = 256;
constexpr uint64_t SPLIT_H_MAX_WOUT = 512;
constexpr uint64_t SPLIT_H_SMALL_KERNEL_LIMIT = 128;
constexpr uint64_t SPLIT_H_DMA_PER_OUTPUT_LIMIT = 90;
// wIn above which fp32 W-down pays more for the transpose than it gains from row reduction,
// so SmallKernel (priority 4) wins instead. Empirically tuned.
constexpr uint64_t SPLIT_H_FP32_WDOWN_WIN_LIMIT = 130;

namespace optiling {

bool AdaptiveAvgPool2dSplitHTiling::IsCapable()
{
    if (!InitComputeBase(computeInfo_)) {
        return false;
    }

    // SplitH targets W-upsampling cases (wOut>wIn) where the H direction is
    // a LARGE-kernel downsampling (hOut<=hIn, kernelHMax>=32). The naive approach
    // does hi-outer row-reuse which, for H downsampling, RMWs outBuf O(hIn) times -> slow.
    // SplitH instead does a ho-driven in-register reduction of each Ho window's covered rows,
    // writing outBuf once per (Ho, straddled-hiBatch) -> outBuf RMW shrinks to O(hIn/hiFactor).
    bool isWSmall = (input_.wIn <= SPLIT_H_WIN_THRESHOLD);
    bool isWUpsampling = (input_.wOut > input_.wIn);
    bool isHDownsampling = (input_.hOut < input_.hIn);
    bool isNcEnough = (input_.nIn * input_.cIn >= computeInfo_.vfLen / TILING_DOUBLE);
    // Kernel keeps wStart/wKerSize in UB buffers sized by wOut; guard wOut.
    bool isWOutBounded = (input_.wOut <= SPLIT_H_MAX_WOUT);
    // H-upsample with low magnification (hOut < 2*hIn), small kernel (kH*kW < 128),
    // and NC < vfLen: SplitH's transpose/RMW overhead dominates. SmallKernel (priority 4)
    // handles this shape class more efficiently with simpler tiling.
    bool isHUpLowMag = (input_.hOut > input_.hIn) && (input_.hOut < TILING_DOUBLE * input_.hIn);
    uint64_t nc = input_.nIn * input_.cIn;
    bool isSmallKernelBetter = isHUpLowMag && (nc < computeInfo_.vfLen) &&
                               (computeInfo_.kernelHMax * computeInfo_.kernelWMax < SPLIT_H_SMALL_KERNEL_LIMIT);
    bool isDmaPerOutputTooHigh = (input_.hIn > SPLIT_H_DMA_PER_OUTPUT_LIMIT * input_.hOut * input_.wOut);
    bool isSingleRowSmallKernel = (input_.hOut <= 1) &&
                                  (computeInfo_.kernelHMax * computeInfo_.kernelWMax < SPLIT_H_SMALL_KERNEL_LIMIT);
    // fp32 W-down with large wIn: SplitH pays a high transpose cost that outweighs the
    // row-reduction benefit. SmallKernel (priority 4) reduces the small (kH*kW<128) window
    // directly and wins on these shapes.
    bool isFp32WDownLargeWin = (computeInfo_.xDtypeSize == sizeof(float)) && (input_.wOut < input_.wIn) &&
                               (input_.wIn >= SPLIT_H_FP32_WDOWN_WIN_LIMIT) &&
                               (computeInfo_.kernelHMax * computeInfo_.kernelWMax < SPLIT_H_SMALL_KERNEL_LIMIT);
    bool isCapable = isWSmall && (isWUpsampling || isHDownsampling) && isNcEnough && isWOutBounded &&
                     !isSmallKernelBetter && !isDmaPerOutputTooHigh && !isSingleRowSmallKernel &&
                     !isFp32WDownLargeWin && IsMeetUbSize();

    OP_LOGD(context_->GetNodeName(),
            "AdaptiveAvgPool2dSplitHTiling IsCapable: kHMax=%lu, kWMax=%lu, wIn=%lu, wOut=%lu, hIn=%lu, hOut=%lu, "
            "NC=%lu, isSmallKernelBetter=%s, isDmaPerOutputTooHigh=%s, isSingleRowSmallKernel=%s, "
            "isFp32WDownLargeWin=%s, result=%s",
            computeInfo_.kernelHMax, computeInfo_.kernelWMax, input_.wIn, input_.wOut, input_.hIn, input_.hOut, nc,
            isSmallKernelBetter ? "true" : "false", isDmaPerOutputTooHigh ? "true" : "false",
            isSingleRowSmallKernel ? "true" : "false", isFp32WDownLargeWin ? "true" : "false",
            isCapable ? "true" : "false");
    return isCapable;
}

void AdaptiveAvgPool2dSplitHTiling::CalUbSplitSize()
{
    if (input_.wOut > input_.wIn) {
        CalCommonUbSplitSize(computeInfo_, input_.wIn);
        return;
    }
    // W↓: SplitH does early Cast(fp32→T) then TransposeB16, so resQue uses sizeof(T).
    // outBuf/resQue use compact wOut instead of wOutAlign to save UB.
    uint64_t wInAlign = Ops::Base::CeilAlign(input_.wIn, computeInfo_.alignNum);
    uint64_t vlNum = computeInfo_.ncFactor;
    uint64_t hoNum = computeInfo_.hoFactor;
    uint64_t hiNum = computeInfo_.hiFactor;
    computeInfo_.inputQueSize = vlNum * hiNum * wInAlign * computeInfo_.xDtypeSize;
    uint64_t outTransAlign = Ops::Base::CeilAlign(hoNum * input_.wOut, TILING_TRANS_ADDR_LEN);
    computeInfo_.resQue1Size = outTransAlign * vlNum * computeInfo_.xDtypeSize;
    computeInfo_.resQue2Size = (computeInfo_.xDtypeSize < sizeof(float)) ?
                                   outTransAlign * vlNum * computeInfo_.xDtypeSize :
                                   0;
}

bool AdaptiveAvgPool2dSplitHTiling::IsMeetUbSize()
{
    CalUbSplitSize();
    uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
    if (input_.wOut > input_.wIn) {
        uint64_t baseOccupy = CalCommonUbOccupy(computeInfo_, input_.wIn);
        uint64_t wiBufSize = Ops::Base::CeilAlign(input_.wIn * sizeof(int32_t), dataBlock) * TILING_DOUBLE;
        // W↑ non-KW1 takes the kernel's two-phase path, which allocates tempSumBuf_ *instead of*
        // sumBuf_ (see AdaptiveAvgPool2dSplitH::Init). CalCommonUbOccupy above already counts
        // sumBuf_ (wOutAlign*ncF*sizeof(float)), so swap it for tempSumBuf_ (wInAlign*ncF*sizeof(float)).
        // Charging both left the search ~40KB short on some W↑ non-KW1 shapes, capping
        // hiFactor at 3 when 6 fits.
        if (input_.wOut % input_.wIn != 0) {
            uint64_t wInAlignUp = Ops::Base::CeilAlign(input_.wIn, computeInfo_.alignNum);
            uint64_t wOutAlignUp = Ops::Base::CeilAlign(input_.wOut, computeInfo_.alignNum);
            uint64_t sumBufSize = wOutAlignUp * computeInfo_.ncFactor * sizeof(float);
            uint64_t tempSumBufSize = wInAlignUp * computeInfo_.ncFactor * sizeof(float);
            return (baseOccupy + wiBufSize + tempSumBufSize) <= (computeInfo_.availableUbSize + sumBufSize);
        }
        return (baseOccupy + wiBufSize) <= computeInfo_.availableUbSize;
    }
    // W↓: compact layout with wOut and sizeof(T) for resQue. No sumBuf (SplitH doesn't use it).
    uint64_t wInAlign = Ops::Base::CeilAlign(input_.wIn, computeInfo_.alignNum);
    uint64_t vlNum = computeInfo_.ncFactor;
    uint64_t hoNum = computeInfo_.hoFactor;
    uint64_t hiNum = computeInfo_.hiFactor;
    uint64_t outTransAlign = Ops::Base::CeilAlign(hoNum * input_.wOut, TILING_TRANS_ADDR_LEN);
    uint64_t transRowAlign = Ops::Base::CeilAlign(hiNum * wInAlign, TILING_TRANS_ADDR_LEN);
    uint64_t transBufSize = transRowAlign * vlNum * computeInfo_.xDtypeSize;
    uint64_t outBufSize = outTransAlign * vlNum * sizeof(float);
    uint64_t wBufSize = Ops::Base::CeilAlign(input_.wOut * sizeof(int32_t), dataBlock) * TILING_DOUBLE;
    // The kernel unconditionally allocates extraWoIdxBuf_ on the W↓ path (see
    // AdaptiveAvgPool2dSplitH::Init); keep this term in sync with that InitBuffer size.
    uint64_t extraWoIdxBufSize = Ops::Base::CeilAlign(input_.wOut * sizeof(int32_t), dataBlock);
    uint64_t totalOccupy = computeInfo_.inputQueSize + computeInfo_.resQue1Size + computeInfo_.resQue2Size +
                           transBufSize + outBufSize + wBufSize + extraWoIdxBufSize;
    return totalOccupy <= computeInfo_.availableUbSize;
}

ge::graphStatus AdaptiveAvgPool2dSplitHTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dSplitHTiling DoOpTiling start.");

    auto meetUb = [this]() { return IsMeetUbSize(); };

    if (input_.wOut <= input_.wIn) {
        // W↓: DMA-call-bound. hiF can exceed kernelHMax (it's just a batch size).
        // Keep ncF=vfLen (full vector utilization) and search hiF upward.
        computeInfo_.ncFactor = computeInfo_.vfLen;
        ShrinkHiFactor(computeInfo_, meetUb);
        BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);

        uint64_t bestHiF = computeInfo_.hiFactor;
        uint64_t bestHoF = computeInfo_.hoFactor;
        CalUbBlockFactor(computeInfo_);
        uint64_t bestScaled = UINT64_MAX;
        uint64_t bestHoOuter = computeInfo_.hoOuter;
        {
            uint64_t totalBatches = 0;
            for (uint64_t hoIdx = 0; hoIdx < computeInfo_.hoOuter; hoIdx++) {
                uint64_t hoStart = hoIdx * bestHoF;
                uint64_t hoEnd = std::min(hoStart + bestHoF, input_.hOut);
                uint64_t hiMin = (hoStart * input_.hIn) / input_.hOut;
                uint64_t hiMax = Ops::Base::CeilDiv(hoEnd * input_.hIn, input_.hOut);
                totalBatches += Ops::Base::CeilDiv(hiMax - hiMin, bestHiF);
            }
            bestScaled = computeInfo_.blockFactor *
                         (totalBatches * TILING_DOUBLE + computeInfo_.hoOuter * TILING_DOUBLE);
        }

        for (uint64_t tryHiF = computeInfo_.kernelHMax + 1; tryHiF <= input_.hIn; tryHiF++) {
            computeInfo_.hiFactor = tryHiF;
            computeInfo_.hoFactor = 1;
            BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);
            if (computeInfo_.hoFactor <= 1) {
                break;
            }
            CalUbBlockFactor(computeInfo_);
            uint64_t hoOuter = computeInfo_.hoOuter;
            uint64_t totalBatches = 0;
            for (uint64_t hoIdx = 0; hoIdx < hoOuter; hoIdx++) {
                uint64_t hoStart = hoIdx * computeInfo_.hoFactor;
                uint64_t hoEnd = std::min(hoStart + computeInfo_.hoFactor, input_.hOut);
                uint64_t hiMin = (hoStart * input_.hIn) / input_.hOut;
                uint64_t hiMax = Ops::Base::CeilDiv(hoEnd * input_.hIn, input_.hOut);
                totalBatches += Ops::Base::CeilDiv(hiMax - hiMin, tryHiF);
            }
            uint64_t scaled = computeInfo_.blockFactor * (totalBatches * TILING_DOUBLE + hoOuter * TILING_DOUBLE);
            if (scaled * bestHoOuter < bestScaled * hoOuter) {
                bestScaled = scaled;
                bestHoOuter = hoOuter;
                bestHiF = tryHiF;
                bestHoF = computeInfo_.hoFactor;
            }
        }
        computeInfo_.hiFactor = bestHiF;
        computeInfo_.hoFactor = bestHoF;

        OptimizeHoFactorForCores(computeInfo_);
    } else {
        StandardUbOptimization(computeInfo_, meetUb);

        // hiFactor is only a DMA batch size, so it may exceed kernelHMax; ShrinkHiFactor above
        // only searches downward from kernelHMax and cannot reach the larger values that cut the
        // batch count. Cost = blockFactor * totalBatches: blockFactor is the per-core workload and
        // totalBatches the DMA calls per block, so the product tracks per-core DMA cost. Both are
        // needed — scoring batches alone picks shapes that shrink hoFactor and inflate hoOuter.
        uint64_t bestHiF = computeInfo_.hiFactor;
        uint64_t bestHoF = computeInfo_.hoFactor;
        uint64_t bestNcF = computeInfo_.ncFactor;
        CalUbBlockFactor(computeInfo_);
        const uint64_t baseUseCoreNum = computeInfo_.useCoreNum;
        uint64_t bestCost = UINT64_MAX;
        for (uint64_t tryHiF = 1; tryHiF <= input_.hIn; tryHiF++) {
            computeInfo_.hiFactor = tryHiF;
            computeInfo_.hoFactor = input_.hOut;
            BinarySearchMaxHoFactor(computeInfo_, input_.hOut, meetUb);
            if (!meetUb()) {
                continue;
            }
            CalUbBlockFactor(computeInfo_);
            // A larger hiFactor costs UB, which shrinks hoFactor and can leave cores idle. Never
            // trade away core occupancy for a lower DMA count.
            if (computeInfo_.useCoreNum < baseUseCoreNum) {
                continue;
            }
            uint64_t totalBatches = 0;
            for (uint64_t hoIdx = 0; hoIdx < computeInfo_.hoOuter; hoIdx++) {
                uint64_t hoStart = hoIdx * computeInfo_.hoFactor;
                uint64_t hoEnd = std::min(hoStart + computeInfo_.hoFactor, input_.hOut);
                uint64_t hiMin = (hoStart * input_.hIn) / input_.hOut;
                uint64_t hiMax = Ops::Base::CeilDiv(hoEnd * input_.hIn, input_.hOut);
                totalBatches += Ops::Base::CeilDiv(hiMax - hiMin, tryHiF);
            }
            uint64_t cost = computeInfo_.blockFactor * totalBatches;
            if (cost < bestCost) {
                bestCost = cost;
                bestHiF = tryHiF;
                bestHoF = computeInfo_.hoFactor;
                bestNcF = computeInfo_.ncFactor;
            }
        }
        computeInfo_.ncFactor = bestNcF;
        computeInfo_.hiFactor = bestHiF;
        computeInfo_.hoFactor = bestHoF;
        CalUbBlockFactor(computeInfo_);
    }

    if (input_.hIn == 1 && computeInfo_.kernelWMax == 1) {
        uint64_t wInAlign = Ops::Base::CeilAlign(input_.wIn, computeInfo_.alignNum);
        uint64_t outTransAlign = Ops::Base::CeilAlign(input_.hOut * input_.wOut, TILING_TRANS_ADDR_LEN);
        uint64_t transRowAlign = Ops::Base::CeilAlign(computeInfo_.hiFactor * wInAlign, TILING_TRANS_ADDR_LEN);
        uint64_t transBufSize = transRowAlign * computeInfo_.ncFactor * computeInfo_.xDtypeSize;
        uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
        uint64_t wiBufSize = Ops::Base::CeilAlign(input_.wIn * sizeof(int32_t), dataBlock) * TILING_DOUBLE;

        uint64_t slimInputQue = computeInfo_.ncFactor * computeInfo_.hiFactor * wInAlign * computeInfo_.xDtypeSize;
        uint64_t slimResQue = outTransAlign * computeInfo_.ncFactor * computeInfo_.xDtypeSize;
        uint64_t slimTotal = slimInputQue + transBufSize + slimResQue * TILING_DOUBLE + wiBufSize;

        if (slimTotal <= computeInfo_.availableUbSize) {
            computeInfo_.hoFactor = input_.hOut;
            computeInfo_.inputQueSize = slimInputQue;
            computeInfo_.resQue1Size = slimResQue;
            computeInfo_.resQue2Size = slimResQue;
            CalUbBlockFactor(computeInfo_);
            OP_LOGD(context_->GetNodeName(), "SplitH slim: hIn=1 kW=1 hoFactor=%lu slimTotal=%lu avail=%lu",
                    computeInfo_.hoFactor, slimTotal, computeInfo_.availableUbSize);
        } else {
            CalUbSplitSize();
        }
    } else {
        CalUbSplitSize();
    }

    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2dSplitHTiling SetTilingData failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dSplitHTiling::SetTilingData()
{
    return FillCommonTilingData<AdaptivePool2dSplitHTilingData>(context_, input_, computeInfo_);
}

void AdaptiveAvgPool2dSplitHTiling::PrintTilingData() const { PrintCommonTilingData(context_, input_, computeInfo_); }

uint64_t AdaptiveAvgPool2dSplitHTiling::GetTilingKey() const
{
    return CalCommonTilingKey(TPL_SPLIT_H_KERNEL, computeInfo_);
}

ge::graphStatus AdaptiveAvgPool2dSplitHTiling::PostTiling()
{
    context_->SetBlockDim(computeInfo_.useCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dSplitHTiling, 1);
} // namespace optiling
