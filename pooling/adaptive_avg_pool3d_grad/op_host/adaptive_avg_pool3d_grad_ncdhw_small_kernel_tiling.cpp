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
 * \file adaptive_avg_pool3d_grad_tiling.cpp
 * \brief
 *
 */

#include "adaptive_avg_pool3d_grad_ncdhw_small_kernel_tiling.h"
#include <algorithm>
#include <sstream>
#include <limits>

namespace optiling {
using namespace AdaptiveAvgPool3dGradOp;

constexpr int64_t TRANS_ADDR_LEN = 16;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t KERNEL_SIZE_MAX = 256;
constexpr int64_t HIGH_THRESHOLD = 128;
constexpr int64_t WINSIZE_THRESHOLD = 16;
constexpr int64_t INPUTW_FLOAT_THRESHOLD = 8;
constexpr int64_t INPUTW_BFLOAT_THRESHOLD = 16;
constexpr int64_t LIMIT = 1;
constexpr int64_t KERNEL_SIZE_SMALL_FP32 = 4;
constexpr int64_t KERNEL_SIZE_SMALL_FP16 = 7;
constexpr int64_t NC_SEARCH_MAX = 256;
constexpr int64_t SEARCH_DHW_SIZE_LIMIT = 200000;

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
constexpr int64_t STRONG_COLLAPSE_RATIO = 3;

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

void AdaptiveAvgPool3dGradTilingSmallKernel::InitializationVars()
{
    gradInputN = inputData.nGrad;
    gradInputC = inputData.cGrad;
    gradInputD = inputData.dGrad;
    gradInputH = inputData.hGrad;
    gradInputW = inputData.wGrad;

    gradOutputN = inputData.nX;
    gradOutputC = inputData.cX;
    gradOutputD = inputData.dX;
    gradOutputH = inputData.hX;
    gradOutputW = inputData.wX;

    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.availableUb = ubSize_ - UB_RESVERVED_SIZE - UB_TEMP_BUFF_SIZE;
    baseData.totalCoreNum = coreNum_;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;
    baseData.maxDataNumInOneBlock = baseData.ubBlockSize / baseData.inputBytes;
    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.inputBytes;
    baseData.inputNCSize = gradOutputN * gradOutputC;
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DoBufferCalculate()
{
    splitData.inputQueBufferSize = 0;
    splitData.transQueBufferSize = 0;
    splitData.transOutQueBufferSize = 0;
    splitData.totalBufferSize = 0;

    const int64_t transRowAlign = TRANS_ADDR_LEN;
    const int64_t transColAlign = baseData.maxDataNumInOneBlock;

    const int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner * gradInputD, gradOutputD) + 1;
    const int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    const int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1;

    const int64_t highAxisInner = splitData.highAxisInner;
    const int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, transColAlign);
    const int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, transColAlign);

    const int64_t inputColNum = dInputInner * hInputInner * wInputInnerAligned;
    const int64_t inputElemNum = highAxisInner * inputColNum;
    const int64_t inputBytes = inputElemNum * baseData.inputBytes;

    const int64_t outputRowNum = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;
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

bool AdaptiveAvgPool3dGradTilingSmallKernel::IsCapable()
{
    InitializationVars();
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }

    kernelD = Ops::Base::CeilDiv(gradOutputD, gradInputD);
    kernelH = Ops::Base::CeilDiv(gradOutputH, gradInputH);
    kernelW = Ops::Base::CeilDiv(gradOutputW, gradInputW);
    if (kernelD * kernelH * kernelW >= KERNEL_SIZE_MAX || baseData.inputNCSize < HIGH_THRESHOLD ||
        gradInputW * gradInputH * gradInputD < WINSIZE_THRESHOLD) {
        return false;
    }

    const int64_t kernelSizeSmall = (inputData.inputDtype == ge::DT_FLOAT) ? KERNEL_SIZE_SMALL_FP32 :
                                                                             KERNEL_SIZE_SMALL_FP16;
    if (gradInputW > gradOutputW && gradOutputD > gradInputD && gradOutputH > gradInputH &&
        kernelD * kernelH * kernelW <= kernelSizeSmall) {
        return false;
    }

    if (inputData.inputDtype == ge::DT_FLOAT) {
        if (gradInputW < INPUTW_FLOAT_THRESHOLD) {
            return false;
        }
    } else {
        if (gradInputW < INPUTW_BFLOAT_THRESHOLD) {
            return false;
        }
    }

    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.dOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = 1;
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool3dGradTilingSmallKernel::IsMeetTargetCoreNum()
{
    const int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    const int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    const int64_t tmpDOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
    const int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpDOutputOuter * tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >=
           baseData.coreUsedForBestPerformance;
}

bool AdaptiveAvgPool3dGradTilingSmallKernel::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool3dGradTilingSmallKernel::TrySplitNC()
{
    splitData.wOutputInner = gradOutputW;
    splitData.hOutputInner = gradOutputH;
    splitData.dOutputInner = gradOutputD;
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;

    return IsMeetUBSize() && IsMeetTargetCoreNum();
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DynamicAdjustmentDWH()
{
    if (splitData.dOutputInner > LIMIT) {
        splitData.dOutputInner = ShrinkInnerStrict(gradOutputD, splitData.dOutputInner);
        return;
    }
    if (splitData.hOutputInner > LIMIT) {
        splitData.hOutputInner = ShrinkInnerStrict(gradOutputH, splitData.hOutputInner);
        return;
    }
    if (splitData.wOutputInner > LIMIT) {
        splitData.wOutputInner = ShrinkInnerStrict(gradOutputW, splitData.wOutputInner);
        return;
    }
}

void AdaptiveAvgPool3dGradTilingSmallKernel::SplitUnalignDHW()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.dOutputInner = gradOutputD;
    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    while (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
        const int64_t oldD = splitData.dOutputInner;
        const int64_t oldH = splitData.hOutputInner;
        const int64_t oldW = splitData.wOutputInner;

        DynamicAdjustmentDWH();

        if (oldD == splitData.dOutputInner && oldH == splitData.hOutputInner && oldW == splitData.wOutputInner) {
            break;
        }
    }

    DoBufferCalculate();
}

void AdaptiveAvgPool3dGradTilingSmallKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }

    const int64_t computeVl = std::max<int64_t>(TRANS_ADDR_LEN, baseData.vRegSize / FLOAT32_SIZE);
    const int64_t inputVl = baseData.proDataNumInOneBeatT2;
    const int64_t searchDhwSize = gradOutputD * gradOutputH * gradOutputW;

    bool found = false;
    int64_t bestHighAxisInner = 0;
    int64_t bestDOutputInner = 0;
    int64_t bestHOutputInner = 0;
    int64_t bestWOutputInner = 0;
    int64_t bestBlockNum = 0;
    int64_t bestUsedCoreNum = 0;
    int64_t bestHighAxisPadding = 0;
    int64_t bestHighAxisTail = 0;
    int64_t bestBufferSize = 0;
    long double bestCost = std::numeric_limits<long double>::max();

    if (searchDhwSize <= SEARCH_DHW_SIZE_LIMIT) {
        int64_t ncSearchMax = std::max<int64_t>(Ops::Base::CeilAlign(baseData.inputNCSize, TRANS_ADDR_LEN),
                                                inputVl * NC_SEARCH_INPUT_VL_MULTIPLIER);
        ncSearchMax = std::min<int64_t>(ncSearchMax, static_cast<int64_t>(NC_SEARCH_MAX));
        ncSearchMax = std::max<int64_t>(ncSearchMax, computeVl);

        ExhaustiveSearchBestTiling(computeVl, ncSearchMax, bestHighAxisInner, bestDOutputInner, bestHOutputInner,
                                   bestWOutputInner, bestBlockNum, bestUsedCoreNum, bestHighAxisPadding,
                                   bestHighAxisTail, bestBufferSize, bestCost, found);
    }

    if (found) {
        splitData.highAxisInner = bestHighAxisInner;
        splitData.dOutputInner = bestDOutputInner;
        splitData.hOutputInner = bestHOutputInner;
        splitData.wOutputInner = bestWOutputInner;
        DoBufferCalculate();
        return;
    }

    ApplyCoarseFallback();
}

bool AdaptiveAvgPool3dGradTilingSmallKernel::ExhaustiveSearchBestTiling(
    int64_t computeVl, int64_t ncSearchMax, int64_t& bestHighAxisInner, int64_t& bestDOutputInner,
    int64_t& bestHOutputInner, int64_t& bestWOutputInner, int64_t& bestBlockNum, int64_t& bestUsedCoreNum,
    int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail, int64_t& bestBufferSize, long double& bestCost,
    bool& found)
{
    for (int64_t highAxisInner = computeVl; highAxisInner <= ncSearchMax; highAxisInner += TRANS_ADDR_LEN) {
        splitData.highAxisInner = highAxisInner;
        const int64_t highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, highAxisInner);
        const int64_t highAxisTail = (baseData.inputNCSize % highAxisInner == 0) ?
                                         highAxisInner :
                                         (baseData.inputNCSize % highAxisInner);
        const int64_t highAxisPadding = highAxisOuter * highAxisInner - baseData.inputNCSize;
        for (int64_t dOutputInner = LIMIT; dOutputInner <= gradOutputD; ++dOutputInner) {
            splitData.dOutputInner = dOutputInner;
            const int64_t dOutputOuter = Ops::Base::CeilDiv(gradOutputD, dOutputInner);
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
                    const int64_t blockNum = highAxisOuter * dOutputOuter * hOutputOuter * wOutputOuter;
                    if (blockNum < baseData.coreUsedForBestPerformance) {
                        continue;
                    }
                    const int64_t normalCoreProcessNum = Ops::Base::CeilDiv(blockNum, baseData.totalCoreNum);
                    long double cost = EvalTilingCandidate(highAxisInner, highAxisOuter, highAxisTail, highAxisPadding,
                                                           dOutputInner, dOutputOuter, hOutputInner, hOutputOuter,
                                                           wOutputInner, wOutputOuter, blockNum, computeVl,
                                                           normalCoreProcessNum);
                    if (cost < 0.0L) {
                        continue;
                    }
                    const int64_t usedCoreNum = Ops::Base::CeilDiv(blockNum, normalCoreProcessNum);
                    TryRecordBetterTiling(cost, dOutputInner, hOutputInner, wOutputInner, blockNum, usedCoreNum,
                                          highAxisInner, highAxisPadding, highAxisTail, bestHighAxisInner,
                                          bestDOutputInner, bestHOutputInner, bestWOutputInner, bestBlockNum,
                                          bestUsedCoreNum, bestHighAxisPadding, bestHighAxisTail, bestBufferSize,
                                          bestCost, found);
                }
            }
        }
    }
    return found;
}

long double AdaptiveAvgPool3dGradTilingSmallKernel::EvalTilingCandidate(
    int64_t highAxisInner, int64_t highAxisOuter, int64_t highAxisTail, int64_t highAxisPadding, int64_t dOutputInner,
    int64_t dOutputOuter, int64_t hOutputInner, int64_t hOutputOuter, int64_t wOutputInner, int64_t wOutputOuter,
    int64_t blockNum, int64_t computeVl, int64_t normalCoreProcessNum)
{
    const int64_t oneBufferSize = splitData.inputQueBufferSize + splitData.transQueBufferSize +
                                  splitData.transOutQueBufferSize;
    const int64_t dInputInner = Ops::Base::CeilDiv(dOutputInner * gradInputD, gradOutputD) + HW_INNER_SAFE_MARGIN;
    const int64_t hInputInner = Ops::Base::CeilDiv(hOutputInner * gradInputH, gradOutputH) + HW_INNER_SAFE_MARGIN;
    const int64_t wInputInner = Ops::Base::CeilDiv(wOutputInner * gradInputW, gradOutputW) + HW_INNER_SAFE_MARGIN;
    const int64_t actualInputElem = highAxisInner * dInputInner * hInputInner * wInputInner;
    const int64_t actualOutputElem = highAxisInner * dOutputInner * hOutputInner * wOutputInner;
    const int64_t oneBlockWork = oneBufferSize + actualInputElem * (baseData.inputBytes + FLOAT32_SIZE) +
                                 actualOutputElem * FLOAT32_SIZE * OUTPUT_FP32_FACTOR +
                                 baseData.ubBlockSize * WORK_PER_BLOCK_UB_OVERHEAD;

    long double cost = static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
    cost += static_cast<long double>(highAxisPadding) * static_cast<long double>(gradOutputD) *
            static_cast<long double>(gradOutputH) * static_cast<long double>(gradOutputW) *
            COST_HIGH_AXIS_PADDING_FACTOR;

    cost = AddCostPenalties(cost, highAxisInner, highAxisOuter, highAxisTail, dOutputInner, hOutputInner, wOutputInner,
                            blockNum, computeVl, normalCoreProcessNum, oneBlockWork);
    return cost;
}

long double AdaptiveAvgPool3dGradTilingSmallKernel::AddCostPenalties(long double cost, int64_t highAxisInner,
                                                                     int64_t highAxisOuter, int64_t highAxisTail,
                                                                     int64_t dOutputInner, int64_t hOutputInner,
                                                                     int64_t wOutputInner, int64_t blockNum,
                                                                     int64_t computeVl, int64_t normalCoreProcessNum,
                                                                     int64_t oneBlockWork)
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
            dOutputInner * hOutputInner * Ops::Base::CeilAlign(wOutputInner, baseData.maxDataNumInOneBlock),
            TRANS_ADDR_LEN);
        cost += static_cast<long double>(blockNum) * static_cast<long double>(alignedOutputRow) *
                static_cast<long double>(highAxisInner) * COST_TRANS_ALIGN_FACTOR;
    }

    const int64_t idleCoreNum = baseData.totalCoreNum - Ops::Base::CeilDiv(blockNum, normalCoreProcessNum);
    cost += static_cast<long double>(std::max<int64_t>(0, idleCoreNum)) * static_cast<long double>(oneBlockWork) *
            COST_IDLE_CORE_FACTOR;

    if (dOutputInner == LIMIT && gradOutputD > LIMIT) {
        cost += static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
    }

    if (hOutputInner == LIMIT && gradOutputH > LIMIT) {
        cost += static_cast<long double>(normalCoreProcessNum) * static_cast<long double>(oneBlockWork);
    }

    return cost;
}

bool AdaptiveAvgPool3dGradTilingSmallKernel::TryRecordBetterTiling(
    long double cost, int64_t dOutputInner, int64_t hOutputInner, int64_t wOutputInner, int64_t blockNum,
    int64_t usedCoreNum, int64_t highAxisInner, int64_t highAxisPadding, int64_t highAxisTail,
    int64_t& bestHighAxisInner, int64_t& bestDOutputInner, int64_t& bestHOutputInner, int64_t& bestWOutputInner,
    int64_t& bestBlockNum, int64_t& bestUsedCoreNum, int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail,
    int64_t& bestBufferSize, long double& bestCost, bool& found)
{
    bool better = false;
    if (!found || cost < bestCost) {
        better = true;
    } else if (cost == bestCost) {
        const int64_t curArea = dOutputInner * hOutputInner * wOutputInner;
        const int64_t bestArea = bestDOutputInner * bestHOutputInner * bestWOutputInner;
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
    bestDOutputInner = dOutputInner;
    bestHOutputInner = hOutputInner;
    bestWOutputInner = wOutputInner;
    bestBlockNum = blockNum;
    bestUsedCoreNum = usedCoreNum;
    bestHighAxisPadding = highAxisPadding;
    bestHighAxisTail = highAxisTail;
    bestBufferSize = splitData.totalBufferSize;
    return true;
}

void AdaptiveAvgPool3dGradTilingSmallKernel::ApplyCoarseFallback()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.dOutputInner = gradOutputD;
    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    while (splitData.dOutputInner > kernelD || splitData.hOutputInner > kernelH || splitData.wOutputInner > kernelW) {
        if (IsMeetTargetCoreNum() && IsMeetUBSize()) {
            return;
        }

        if (splitData.dOutputInner > kernelD) {
            splitData.dOutputInner -= kernelD;
            continue;
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

    SplitUnalignDHW();
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    splitData.wOutputTail = (gradOutputW % splitData.wOutputInner == 0) ? splitData.wOutputInner :
                                                                          (gradOutputW % splitData.wOutputInner);

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.hOutputTail = (gradOutputH % splitData.hOutputInner == 0) ? splitData.hOutputInner :
                                                                          (gradOutputH % splitData.hOutputInner);

    splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
    splitData.dOutputTail = (gradOutputD % splitData.dOutputInner == 0) ? splitData.dOutputInner :
                                                                          (gradOutputD % splitData.dOutputInner);

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    splitData.highAxisTail = (baseData.inputNCSize % splitData.highAxisInner == 0) ?
                                 splitData.highAxisInner :
                                 (baseData.inputNCSize % splitData.highAxisInner);
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter *
                                  splitData.dOutputOuter;

    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void AdaptiveAvgPool3dGradTilingSmallKernel::SetTilingData()
{
    tilingData->dInput = gradInputD;
    tilingData->hInput = gradInputH;
    tilingData->wInput = gradInputW;
    tilingData->dOutput = gradOutputD;
    tilingData->hOutput = gradOutputH;
    tilingData->wOutput = gradOutputW;
    tilingData->highAxisInner = splitData.highAxisInner;
    tilingData->highAxisTail = splitData.highAxisTail;
    tilingData->highAxisOuter = splitData.highAxisOuter;
    tilingData->dOutputInner = splitData.dOutputInner;
    tilingData->dOutputTail = splitData.dOutputTail;
    tilingData->dOutputOuter = splitData.dOutputOuter;
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

void AdaptiveAvgPool3dGradTilingSmallKernel::PrintSplitData() const
{
    OP_LOGD("AdaptiveAvgPool3dGradNCDHW", "[AdaptiveAvgPool3dGradNCDHW] PrintSplitData start running");
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

    std::ostringstream info;
    info << "baseData.availableUb: " << baseData.availableUb << ", inputNCSize: " << baseData.inputNCSize
         << ", totalCoreNum: " << baseData.totalCoreNum << std::endl;

    info << "splitData.highAxisInner: " << splitData.highAxisInner << std::endl;
    info << "splitData.highAxisTail: " << splitData.highAxisTail << std::endl;
    info << "splitData.highAxisOuter: " << splitData.highAxisOuter << std::endl;

    info << "splitData.dOutputInner: " << splitData.dOutputInner << std::endl;
    info << "splitData.dOutputTail: " << splitData.dOutputTail << std::endl;
    info << "splitData.dOutputOuter: " << splitData.dOutputOuter << std::endl;

    info << "splitData.hOutputInner: " << splitData.hOutputInner << std::endl;
    info << "splitData.hOutputTail: " << splitData.hOutputTail << std::endl;
    info << "splitData.hOutputOuter: " << splitData.hOutputOuter << std::endl;

    info << "splitData.wOutputInner: " << splitData.wOutputInner << std::endl;
    info << "splitData.wOutputTail: " << splitData.wOutputTail << std::endl;
    info << "splitData.wOutputOuter: " << splitData.wOutputOuter << std::endl;

    info << "splitData.normalCoreProcessNum: " << splitData.normalCoreProcessNum << std::endl;
    info << "splitData.tailCoreProcessNum: " << splitData.tailCoreProcessNum << std::endl;
    info << "splitData.usedCoreNum: " << splitData.usedCoreNum << std::endl;
    info << "splitData.totalBaseBlockNum: " << splitData.totalBaseBlockNum << std::endl;

    info << "splitData.inputQueBufferSize: " << splitData.inputQueBufferSize << std::endl;
    info << "splitData.transQueBufferSize: " << splitData.transQueBufferSize << std::endl;
    info << "splitData.transOutQueBufferSize: " << splitData.transOutQueBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

    info << "highAxisPadding: " << highAxisPadding << ", highAxisValidRate: " << highAxisValidRate
         << ", ubUseRate: " << ubUseRate << ", coreUseRate: " << coreUseRate << std::endl;

    OP_LOGI("AdaptiveAvgPool3dGradNCDHW", "%s", info.str().c_str());
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSmallKernel::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSmallKernel::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool3dGradTilingSmallKernel::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, idxDtype, isChannelLast);
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSmallKernel::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingSmallKernel::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingSmallKernel, 20);
} // namespace optiling
