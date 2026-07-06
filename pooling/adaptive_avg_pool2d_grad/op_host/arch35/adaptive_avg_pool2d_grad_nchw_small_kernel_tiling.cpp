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
 * \file adaptive_avg_pool2d_grad_nchw_small_kernel_tiling.cpp
 * \brief
 */

#include "adaptive_avg_pool2d_grad_nchw_small_kernel_tiling.h"
#include <algorithm>
#include <sstream>
#include <limits>

namespace optiling {
using namespace AdaptiveAvgPool2dGradOp;

constexpr int64_t TRANS_ADDR_LEN = 16;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t KERNEL_SIZE_MAX = 256;
constexpr int64_t HIGH_THRESHOLD = 128;
constexpr int64_t WINSIZE_THRESHOLD = 16;
constexpr int64_t INPUTW_FLOAT_THRESHOLD = 8;
constexpr int64_t INPUTW_BFLOAT_THRESHOLD = 16;
constexpr int64_t LIMIT = 1;
constexpr int64_t NC_SEARCH_MAX = 256;
constexpr int64_t SEARCH_HW_SIZE_LIMIT = 200000;

constexpr int64_t PREFER_SIMT_W_THRESHOLD = 4;
constexpr int64_t PREFER_SIMT_H_LOW_THRESHOLD = 64;
constexpr int64_t STRONG_RESIZE_RATIO = 4;
constexpr int64_t UPSAMPLE_AREA_EXPAND_RATIO = 3;
constexpr int64_t STRONG_COLLAPSE_RATIO = 3;
constexpr int64_t RESIZE_W_EXPAND_RATIO = 2;
constexpr int64_t NC_SEARCH_INPUT_VL_MULTIPLIER = 2;
constexpr int64_t HW_INNER_SAFE_MARGIN = 1;
constexpr int64_t OUTPUT_FP32_FACTOR = 2;
constexpr int64_t WORK_PER_BLOCK_UB_OVERHEAD = 64;
constexpr long double COST_HIGH_AXIS_PADDING_FACTOR = 4.0L;
constexpr int64_t HIGH_AXIS_TAIL_OPT_THRESHOLD = 8;
constexpr long double COST_PARTIAL_VL_PENALTY_HW = 1024.0L;
constexpr long double COST_PARTIAL_VL_PENALTY_BASELINE = 2048.0L;
constexpr long double COST_TRANS_ALIGN_FACTOR = 0.25L;
constexpr long double COST_IDLE_CORE_FACTOR = 0.15L;

static inline int64_t ShrinkInnerStrict(int64_t total, int64_t curInner)
{
    if (curInner <= LIMIT) {
        return LIMIT;
    }

    const int64_t curOuter = Ops::Base::CeilDiv(total, curInner);
    int64_t nextInner = Ops::Base::CeilDiv(total, curOuter + 1);

    if (nextInner >= curInner) {
        nextInner = curInner - 1;
    }

    return std::max<int64_t>(static_cast<int64_t>(LIMIT), nextInner);
}

void AdaptiveAvgPool2dGradTilingSmallKernel::InitializationVars()
{
    gradInputN = inputData_.nGrad;
    gradInputC = inputData_.cGrad;
    gradInputH = inputData_.hGrad;
    gradInputW = inputData_.wGrad;

    gradOutputN = inputData_.nX;
    gradOutputC = inputData_.cX;
    gradOutputH = inputData_.hX;
    gradOutputW = inputData_.wX;

    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData_.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.availableUb = ubSize_ - UB_RESVERVED_SIZE - UB_TEMP_BUFF_SIZE;
    baseData.totalCoreNum = coreNum_;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;
    baseData.maxDataNumInOneBlock = baseData.ubBlockSize / baseData.inputBytes;
    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.inputBytes;
    baseData.inputNCSize = gradOutputN * gradOutputC;

    const int64_t computeVl = baseData.vRegSize / FLOAT32_SIZE;
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoBufferCalculate()
{
    splitData.inputQueBufferSize = 0;
    splitData.transQueBufferSize = 0;
    splitData.transOutQueBufferSize = 0;
    splitData.totalBufferSize = 0;

    const int64_t transRowAlign = TRANS_ADDR_LEN;
    const int64_t transColAlign = baseData.maxDataNumInOneBlock;

    const int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    const int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1;

    const int64_t highAxisInner = splitData.highAxisInner;
    const int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, transColAlign);
    const int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, transColAlign);

    const int64_t inputColNum = hInputInner * wInputInnerAligned;
    const int64_t inputElemNum = highAxisInner * inputColNum;
    const int64_t inputBytes = inputElemNum * baseData.inputBytes;

    const int64_t outputRowNum = splitData.hOutputInner * wOutputInnerAligned;
    const int64_t outputRowNumAligned = Ops::Base::CeilAlign(outputRowNum, transRowAlign);
    const int64_t outputElemNum = outputRowNumAligned * highAxisInner;

    const int64_t transQueBytes = std::max(inputElemNum, outputElemNum) * FLOAT32_SIZE;
    const int64_t transOutQueBytes = outputElemNum * FLOAT32_SIZE;

    splitData.inputQueBufferSize = Ops::Base::CeilAlign(inputBytes, baseData.ubBlockSize);
    splitData.transQueBufferSize = Ops::Base::CeilAlign(transQueBytes, baseData.ubBlockSize);
    splitData.transOutQueBufferSize = Ops::Base::CeilAlign(transOutQueBytes, baseData.ubBlockSize);

    splitData.totalBufferSize = BUFFER_NUM * (splitData.inputQueBufferSize + splitData.transQueBufferSize +
                                              splitData.transOutQueBufferSize);
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::IsCapable()
{
    InitializationVars();

    const bool preferSimtInputAsGradY = gradInputH > gradOutputH && gradInputW <= gradOutputW &&
                                        gradInputW <= PREFER_SIMT_W_THRESHOLD &&
                                        (gradInputH >= gradOutputH * STRONG_RESIZE_RATIO ||
                                         gradOutputH <= PREFER_SIMT_H_LOW_THRESHOLD);

    const bool preferSimtOutputAsGradY = gradOutputH > gradInputH && gradOutputW <= gradInputW &&
                                         gradOutputW <= PREFER_SIMT_W_THRESHOLD &&
                                         (gradOutputH >= gradInputH * STRONG_RESIZE_RATIO ||
                                          gradInputH <= PREFER_SIMT_H_LOW_THRESHOLD);

    if (preferSimtInputAsGradY || preferSimtOutputAsGradY) {
        return false;
    }

    kernelH = Ops::Base::CeilDiv(gradOutputH, gradInputH);
    kernelW = Ops::Base::CeilDiv(gradOutputW, gradInputW);

    const int64_t kernelSize = kernelH * kernelW;
    const int64_t inputWinSize = gradInputW * gradInputH;
    const int64_t highAxis = baseData.inputNCSize;
    const int64_t inputHW = gradInputH * gradInputW;
    const int64_t outputHW = gradOutputH * gradOutputW;

    if (kernelSize >= KERNEL_SIZE_MAX) {
        return false;
    }

    if (baseData.inputNCSize < HIGH_THRESHOLD) {
        return false;
    }

    if (inputWinSize < WINSIZE_THRESHOLD) {
        return false;
    }

    if (inputData_.inputDtype == ge::DT_FLOAT) {
        if (gradInputW < INPUTW_FLOAT_THRESHOLD) {
            return false;
        }
    } else {
        if (gradInputW < INPUTW_BFLOAT_THRESHOLD) {
            return false;
        }
    }

    constexpr int64_t SIMT_RESIZE_KERNEL_SIZE_MAX = 7;
    constexpr int64_t SIMT_RESIZE_NC_MAX = 4096;
    constexpr int64_t SIMT_BOTH_UPSAMPLE_NC_MAX = 700;

    const bool hUpsample = gradOutputH > gradInputH;
    const bool wUpsample = gradOutputW > gradInputW;
    const bool hDownsample = gradOutputH < gradInputH;
    const bool wDownsample = gradOutputW < gradInputW;
    const bool hResize = gradOutputH != gradInputH;
    const bool wResize = gradOutputW != gradInputW;
    const bool twoAxisResize = hResize && wResize;

    const bool bothUpsampleAreaExpand = hUpsample && wUpsample && highAxis <= SIMT_BOTH_UPSAMPLE_NC_MAX &&
                                        outputHW >= inputHW * UPSAMPLE_AREA_EXPAND_RATIO;

    const bool hExpandWCollapseStrong = hUpsample && wDownsample && gradOutputH >= gradInputH * STRONG_RESIZE_RATIO &&
                                        gradInputW >= gradOutputW * STRONG_COLLAPSE_RATIO;

    const bool hCollapseWExpandStrong = hDownsample && wUpsample && gradInputH >= gradOutputH * STRONG_COLLAPSE_RATIO &&
                                        gradOutputW * RESIZE_W_EXPAND_RATIO >= gradInputW * STRONG_COLLAPSE_RATIO;

    const bool preferSimtUnfriendlyResize = kernelSize <= SIMT_RESIZE_KERNEL_SIZE_MAX && highAxis >= HIGH_THRESHOLD &&
                                            highAxis <= SIMT_RESIZE_NC_MAX && twoAxisResize &&
                                            (bothUpsampleAreaExpand || hExpandWCollapseStrong ||
                                             hCollapseWExpandStrong);

    if (preferSimtUnfriendlyResize) {
        return false;
    }

    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.hOutputInner = LIMIT;
    splitData.wOutputInner = LIMIT;
    DoBufferCalculate();

    const bool capable = splitData.totalBufferSize <= baseData.availableUb;
    return capable;
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::IsMeetTargetCoreNum()
{
    const int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    const int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    const int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::TrySplitNC()
{
    splitData.wOutputInner = gradOutputW;
    splitData.hOutputInner = gradOutputH;
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;

    return IsMeetUBSize() && IsMeetTargetCoreNum();
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DynamicAdjustmentHW()
{
    if (splitData.hOutputInner > LIMIT) {
        splitData.hOutputInner = ShrinkInnerStrict(gradOutputH, splitData.hOutputInner);
        return;
    }

    if (splitData.wOutputInner > LIMIT) {
        splitData.wOutputInner = ShrinkInnerStrict(gradOutputW, splitData.wOutputInner);
        return;
    }
}

void AdaptiveAvgPool2dGradTilingSmallKernel::SplitUnalignHW()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    while (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
        const int64_t oldH = splitData.hOutputInner;
        const int64_t oldW = splitData.wOutputInner;

        DynamicAdjustmentHW();

        if (oldH == splitData.hOutputInner && oldW == splitData.wOutputInner) {
            break;
        }
    }

    DoBufferCalculate();
}

void AdaptiveAvgPool2dGradTilingSmallKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }

    const int64_t computeVl = std::max<int64_t>(TRANS_ADDR_LEN, baseData.vRegSize / FLOAT32_SIZE);
    const int64_t inputVl = baseData.proDataNumInOneBeatT2;
    const int64_t searchHwSize = gradOutputH * gradOutputW;

    bool found = false;
    int64_t bestHighAxisInner = 0;
    int64_t bestHOutputInner = 0;
    int64_t bestWOutputInner = 0;
    int64_t bestBlockNum = 0;
    int64_t bestUsedCoreNum = 0;
    int64_t bestHighAxisPadding = 0;
    int64_t bestHighAxisTail = 0;
    int64_t bestBufferSize = 0;
    long double bestCost = std::numeric_limits<long double>::max();

    if (searchHwSize <= SEARCH_HW_SIZE_LIMIT) {
        int64_t ncSearchMax = std::max<int64_t>(Ops::Base::CeilAlign(baseData.inputNCSize, TRANS_ADDR_LEN),
                                                inputVl * NC_SEARCH_INPUT_VL_MULTIPLIER);
        ncSearchMax = std::min<int64_t>(ncSearchMax, static_cast<int64_t>(NC_SEARCH_MAX));
        ncSearchMax = std::max<int64_t>(ncSearchMax, computeVl);

        ExhaustiveSearchBestTiling(computeVl, ncSearchMax, bestHighAxisInner, bestHOutputInner, bestWOutputInner,
                                   bestBlockNum, bestUsedCoreNum, bestHighAxisPadding, bestHighAxisTail, bestBufferSize,
                                   bestCost, found);
    }

    if (found) {
        splitData.highAxisInner = bestHighAxisInner;
        splitData.hOutputInner = bestHOutputInner;
        splitData.wOutputInner = bestWOutputInner;
        DoBufferCalculate();
        return;
    }

    ApplyCoarseFallback();
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::ExhaustiveSearchBestTiling(
    int64_t computeVl, int64_t ncSearchMax, int64_t& bestHighAxisInner, int64_t& bestHOutputInner,
    int64_t& bestWOutputInner, int64_t& bestBlockNum, int64_t& bestUsedCoreNum, int64_t& bestHighAxisPadding,
    int64_t& bestHighAxisTail, int64_t& bestBufferSize, long double& bestCost, bool& found)
{
    for (int64_t highAxisInner = computeVl; highAxisInner <= ncSearchMax; highAxisInner += TRANS_ADDR_LEN) {
        splitData.highAxisInner = highAxisInner;
        const int64_t highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, highAxisInner);
        const int64_t highAxisTail = (baseData.inputNCSize % highAxisInner == 0) ?
                                         highAxisInner :
                                         (baseData.inputNCSize % highAxisInner);
        const int64_t highAxisPadding = highAxisOuter * highAxisInner - baseData.inputNCSize;
        for (int64_t hOutputInner = LIMIT; hOutputInner <= gradOutputH; ++hOutputInner) {
            splitData.hOutputInner = hOutputInner;
            const int64_t hOutputOuter = Ops::Base::CeilDiv(gradOutputH, hOutputInner);
            for (int64_t wOutputInner = LIMIT; wOutputInner <= gradOutputW; ++wOutputInner) {
                splitData.wOutputInner = wOutputInner;
                DoBufferCalculate();
                if (splitData.totalBufferSize > baseData.availableUb) {
                    continue;
                }
                const int64_t wOutputOuter = Ops::Base::CeilDiv(gradOutputW, wOutputInner);
                const int64_t blockNum = highAxisOuter * hOutputOuter * wOutputOuter;
                if (blockNum < baseData.coreUsedForBestPerformance) {
                    continue;
                }
                const int64_t normalCoreProcessNum = Ops::Base::CeilDiv(blockNum, baseData.totalCoreNum);
                long double cost = EvalTilingCandidate(highAxisInner, highAxisOuter, highAxisTail, highAxisPadding,
                                                       hOutputInner, hOutputOuter, wOutputInner, wOutputOuter, blockNum,
                                                       computeVl, normalCoreProcessNum);
                if (cost < 0.0L) {
                    continue;
                }
                const int64_t usedCoreNum = Ops::Base::CeilDiv(blockNum, normalCoreProcessNum);
                TryRecordBetterTiling(cost, hOutputInner, wOutputInner, blockNum, usedCoreNum, highAxisInner,
                                      highAxisPadding, highAxisTail, bestHighAxisInner, bestHOutputInner,
                                      bestWOutputInner, bestBlockNum, bestUsedCoreNum, bestHighAxisPadding,
                                      bestHighAxisTail, bestBufferSize, bestCost, found);
            }
        }
    }
    return found;
}

long double AdaptiveAvgPool2dGradTilingSmallKernel::EvalTilingCandidate(int64_t highAxisInner, int64_t highAxisOuter,
                                                                        int64_t highAxisTail, int64_t highAxisPadding,
                                                                        int64_t hOutputInner, int64_t hOutputOuter,
                                                                        int64_t wOutputInner, int64_t wOutputOuter,
                                                                        int64_t blockNum, int64_t computeVl,
                                                                        int64_t normalCoreProcessNum)
{
    const int64_t oneBufferSize = splitData.inputQueBufferSize + splitData.transQueBufferSize +
                                  splitData.transOutQueBufferSize;
    const int64_t hInputInner = Ops::Base::CeilDiv(hOutputInner * gradInputH, gradOutputH) + HW_INNER_SAFE_MARGIN;
    const int64_t wInputInner = Ops::Base::CeilDiv(wOutputInner * gradInputW, gradOutputW) + HW_INNER_SAFE_MARGIN;
    const int64_t actualInputElem = highAxisInner * hInputInner * wInputInner;
    const int64_t actualOutputElem = highAxisInner * hOutputInner * wOutputInner;
    const int64_t oneBlockWork = oneBufferSize + actualInputElem * (baseData.inputBytes + FLOAT32_SIZE) +
                                 actualOutputElem * FLOAT32_SIZE * OUTPUT_FP32_FACTOR +
                                 baseData.ubBlockSize * WORK_PER_BLOCK_UB_OVERHEAD;

    long double cost = static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
    cost += static_cast<long double>(highAxisPadding) * static_cast<long double>(gradOutputH) *
            static_cast<long double>(gradOutputW) * COST_HIGH_AXIS_PADDING_FACTOR;

    cost = AddCostPenalties(cost, highAxisInner, highAxisOuter, highAxisTail, hOutputInner, wOutputInner, blockNum,
                            computeVl, normalCoreProcessNum, oneBlockWork);
    return cost;
}

long double AdaptiveAvgPool2dGradTilingSmallKernel::AddCostPenalties(long double cost, int64_t highAxisInner,
                                                                     int64_t highAxisOuter, int64_t highAxisTail,
                                                                     int64_t hOutputInner, int64_t wOutputInner,
                                                                     int64_t blockNum, int64_t computeVl,
                                                                     int64_t normalCoreProcessNum, int64_t oneBlockWork)
{
    if (highAxisOuter > 1 && highAxisTail < computeVl) {
        if (highAxisOuter >= HIGH_AXIS_TAIL_OPT_THRESHOLD && gradInputW > gradOutputW) {
            cost += static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork) /
                    static_cast<long double>(highAxisOuter);
        } else {
            cost += static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
        }
    }

    if (computeVl > 0 && highAxisInner % computeVl != 0) {
        const long double partialVlPenalty = gradInputW > gradOutputW ? COST_PARTIAL_VL_PENALTY_HW :
                                                                        COST_PARTIAL_VL_PENALTY_BASELINE;
        cost += static_cast<long double>(highAxisInner % computeVl) * static_cast<long double>(normalCoreProcessNum) *
                partialVlPenalty;
    }

    if (gradInputW > gradOutputW && gradInputW >= gradOutputW * STRONG_COLLAPSE_RATIO) {
        const int64_t alignedOutputRow = Ops::Base::CeilAlign(
            hOutputInner * Ops::Base::CeilAlign(wOutputInner, baseData.maxDataNumInOneBlock), TRANS_ADDR_LEN);
        cost += static_cast<long double>(blockNum) * static_cast<long double>(alignedOutputRow) *
                static_cast<long double>(highAxisInner) * COST_TRANS_ALIGN_FACTOR;
    }

    const int64_t idleCoreNum = baseData.totalCoreNum - Ops::Base::CeilDiv(blockNum, normalCoreProcessNum);
    cost += static_cast<long double>(std::max<int64_t>(0, idleCoreNum)) * static_cast<long double>(oneBlockWork) *
            COST_IDLE_CORE_FACTOR;

    if (hOutputInner == LIMIT && gradOutputH > LIMIT) {
        cost += static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
    }

    return cost;
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::TryRecordBetterTiling(
    long double cost, int64_t hOutputInner, int64_t wOutputInner, int64_t blockNum, int64_t usedCoreNum,
    int64_t highAxisInner, int64_t highAxisPadding, int64_t highAxisTail, int64_t& bestHighAxisInner,
    int64_t& bestHOutputInner, int64_t& bestWOutputInner, int64_t& bestBlockNum, int64_t& bestUsedCoreNum,
    int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail, int64_t& bestBufferSize, long double& bestCost,
    bool& found)
{
    bool better = false;
    if (!found || cost < bestCost) {
        better = true;
    } else if (cost == bestCost) {
        const int64_t curArea = hOutputInner * wOutputInner;
        const int64_t bestArea = bestHOutputInner * bestWOutputInner;
        if (blockNum < bestBlockNum || (blockNum == bestBlockNum && curArea > bestArea) ||
            (blockNum == bestBlockNum && curArea == bestArea && highAxisPadding < bestHighAxisPadding)) {
            better = true;
        }
    }

    if (!better) {
        return false;
    }

    found = true;
    bestCost = cost;
    bestHighAxisInner = highAxisInner;
    bestHOutputInner = hOutputInner;
    bestWOutputInner = wOutputInner;
    bestBlockNum = blockNum;
    bestUsedCoreNum = usedCoreNum;
    bestHighAxisPadding = highAxisPadding;
    bestHighAxisTail = highAxisTail;
    bestBufferSize = splitData.totalBufferSize;
    return true;
}

void AdaptiveAvgPool2dGradTilingSmallKernel::ApplyCoarseFallback()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    while (splitData.hOutputInner > kernelH || splitData.wOutputInner > kernelW) {
        if (IsMeetTargetCoreNum() && IsMeetUBSize()) {
            return;
        }

        if (splitData.hOutputInner > kernelH) {
            splitData.hOutputInner -= kernelH;
            continue;
        }

        if (splitData.wOutputInner > kernelW) {
            splitData.wOutputInner -= kernelW;
            continue;
        }
    }

    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return;
    }

    SplitUnalignHW();
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    splitData.wOutputTail = (gradOutputW % splitData.wOutputInner == 0) ? splitData.wOutputInner :
                                                                          (gradOutputW % splitData.wOutputInner);

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.hOutputTail = (gradOutputH % splitData.hOutputInner == 0) ? splitData.hOutputInner :
                                                                          (gradOutputH % splitData.hOutputInner);

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    splitData.highAxisTail = (baseData.inputNCSize % splitData.highAxisInner == 0) ?
                                 splitData.highAxisInner :
                                 (baseData.inputNCSize % splitData.highAxisInner);
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter;

    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void AdaptiveAvgPool2dGradTilingSmallKernel::SetTilingData()
{
    tilingData->hInput = gradInputH;
    tilingData->wInput = gradInputW;
    tilingData->hOutput = gradOutputH;
    tilingData->wOutput = gradOutputW;

    tilingData->highAxisInner = splitData.highAxisInner;
    tilingData->highAxisTail = splitData.highAxisTail;
    tilingData->highAxisOuter = splitData.highAxisOuter;

    tilingData->hOutputInner = splitData.hOutputInner;
    tilingData->hOutputTail = splitData.hOutputTail;
    tilingData->hOutputOuter = splitData.hOutputOuter;

    tilingData->wOutputInner = splitData.wOutputInner;
    tilingData->wOutputTail = splitData.wOutputTail;
    tilingData->wOutputOuter = splitData.wOutputOuter;

    tilingData->normalCoreProcessNum = splitData.normalCoreProcessNum;
    tilingData->tailCoreProcessNum = splitData.tailCoreProcessNum;
    tilingData->usedCoreNum = splitData.usedCoreNum;

    tilingData->inputQueBufferSize = splitData.inputQueBufferSize;
    tilingData->transQueBufferSize = splitData.transQueBufferSize;
    tilingData->transOutQueBufferSize = splitData.transOutQueBufferSize;
}

void AdaptiveAvgPool2dGradTilingSmallKernel::PrintSplitData() const
{
    const int64_t highAxisTotalCapacity = splitData.highAxisOuter * splitData.highAxisInner;
    const int64_t highAxisPadding = highAxisTotalCapacity - baseData.inputNCSize;
    const double highAxisValidRate = highAxisTotalCapacity == 0 ? 0.0 :
                                                                  static_cast<double>(baseData.inputNCSize) /
                                                                      static_cast<double>(highAxisTotalCapacity);
    const double ubUseRate = baseData.availableUb == 0 ? 0.0 :
                                                         static_cast<double>(splitData.totalBufferSize) /
                                                             static_cast<double>(baseData.availableUb);
    const double coreUseRate = baseData.totalCoreNum == 0 ? 0.0 :
                                                            static_cast<double>(splitData.usedCoreNum) /
                                                                static_cast<double>(baseData.totalCoreNum);
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSmallKernel::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSmallKernel::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool2dGradTilingSmallKernel::GetTilingKey() const
{
    const int64_t outDataCount = inputData_.nX * inputData_.cX * inputData_.hX * inputData_.wX;
    const uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    const uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, idxDtype, isChannelLast);
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSmallKernel::PostTiling()
{
    const uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSmallKernel::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradTilingSmallKernel, 20);
} // namespace optiling
