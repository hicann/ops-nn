
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "pool3d_grad_ndhwc_small_kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESERVED_SIZE = 5120;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

void Pool3DGradNDHWCSmallKernelCommonTiling::InitializationVars(gert::TilingContext* context, int64_t ubSize,
                                                                int64_t coreNum)
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context);
    baseData.inputBytes = inputData->inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.indexBytes = INT32_SIZE;
    baseData.availableUb = ubSize - UB_RESERVED_SIZE;
    baseData.totalCoreNum = coreNum;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;
    int64_t oneBlockNumT2 = baseData.ubBlockSize / baseData.indexBytes;
    baseData.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);
    baseData.proDataNumInOneBeat = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT2;
    baseData.moveDataNumCacheLine = CACHE_LINE_SIZE / baseData.inputBytes;

    baseData.isPad = 0;
    if (inputData->dPad != 0 || inputData->hPad != 0 || inputData->wPad != 0 || inputData->dPadBack != 0 ||
        inputData->hPadBack != 0 || inputData->wPadBack != 0) {
        baseData.isPad = 1;
    }

    baseData.dProBatchSize = 1;
    if (inputData->dKernel > inputData->dStride) {
        baseData.dProBatchSize = Ops::Base::CeilDiv(inputData->dKernel, inputData->dStride);
    }
    baseData.hProBatchSize = 1;
    if (inputData->hKernel > inputData->hStride) {
        baseData.hProBatchSize = Ops::Base::CeilDiv(inputData->hKernel, inputData->hStride);
    }
    baseData.wProBatchSize = 1;
    if (inputData->wKernel > inputData->wStride) {
        baseData.wProBatchSize = Ops::Base::CeilDiv(inputData->wKernel, inputData->wStride);
    }

    baseData.isOverlap = 0;
    if (baseData.dProBatchSize != 1 || baseData.hProBatchSize != 1 || baseData.wProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::IsMeetTargetCoreNum() const
{
    int64_t wOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t hOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t dOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);
    int64_t nOuter = Ops::Base::CeilDiv(inputData->nX, splitData.nOutputInner);
    int64_t cOuter = Ops::Base::CeilDiv(inputData->cX, splitData.cOutputInner);
    return wOuter * hOuter * dOuter * nOuter * cOuter >= baseData.coreUsedForBestPerformance;
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::TrySplitN()
{
    splitData.wOutputInner = inputData->wX;
    splitData.hOutputInner = inputData->hX;
    splitData.dOutputInner = inputData->dX;
    splitData.cOutputInner = inputData->cX;

    splitData.nOutputInner = Ops::Base::CeilDiv(inputData->nX, baseData.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.nOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = inputData->nX;
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.nOutputInner = mid;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.nOutputInner = bestSplit;
        return true;
    }
    return false;
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::TrySplitAlignD()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData->hX;
    splitData.wOutputInner = inputData->wX;
    splitData.cOutputInner = inputData->cX;

    splitData.dOutputInner = inputData->dStride;
    int64_t halfInput = inputData->dX / 2;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(halfInput, inputData->dStride);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.dOutputInner = mid * inputData->dStride;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.dOutputInner = bestSplit * inputData->dStride;
        return true;
    }
    return false;
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::TrySplitAlignH()
{
    splitData.nOutputInner = 1;
    splitData.dOutputInner = inputData->dX;
    splitData.wOutputInner = inputData->wX;
    splitData.cOutputInner = inputData->cX;

    splitData.hOutputInner = inputData->hStride;
    int64_t halfInput = inputData->hX / 2;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(halfInput, inputData->hStride);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.hOutputInner = mid * inputData->hStride;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.hOutputInner = bestSplit * inputData->hStride;
        return true;
    }
    return false;
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::TrySplitAlignW()
{
    splitData.nOutputInner = 1;
    splitData.dOutputInner = inputData->dX;
    splitData.hOutputInner = inputData->hX;
    splitData.cOutputInner = inputData->cX;

    splitData.wOutputInner = inputData->wStride;
    int64_t halfInput = inputData->wX / 2;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(halfInput, inputData->wStride);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.wOutputInner = mid * inputData->wStride;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.wOutputInner = bestSplit * inputData->wStride;
        return true;
    }
    return false;
}

bool Pool3DGradNDHWCSmallKernelCommonTiling::TrySplitAlignC()
{
    splitData.nOutputInner = 1;
    splitData.dOutputInner = inputData->dStride;
    splitData.hOutputInner = inputData->hStride;
    splitData.wOutputInner = inputData->wStride;

    int64_t tmpCAligned = (inputData->cX < baseData.moveDataNumCacheLine) ? inputData->cX :
                                                                            baseData.moveDataNumCacheLine;
    splitData.cOutputInner = tmpCAligned;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->cX / 2, baseData.moveDataNumCacheLine);
        int64_t bestInner = tmpCAligned;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.cOutputInner = mid * baseData.moveDataNumCacheLine;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestInner = splitData.cOutputInner;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.cOutputInner = std::min(bestInner, inputData->cX);
        if (splitData.cOutputInner > baseData.proDataNumInOneBeat) {
            splitData.cOutputInner = baseData.proDataNumInOneBeat;
        }
        return true;
    }
    return false;
}

void Pool3DGradNDHWCSmallKernelCommonTiling::DynamicAdjustmentDHW()
{
    if (splitData.dOutputInner != 1) {
        splitData.dOutputOuter++;
        splitData.dOutputInner = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputOuter);
        return;
    }
    if (splitData.hOutputInner != 1) {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputOuter);
        return;
    }
    splitData.wOutputOuter++;
    splitData.wOutputInner = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputOuter);
}

void Pool3DGradNDHWCSmallKernelCommonTiling::SplitUnalignDHWC()
{
    splitData.nOutputInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.dOutputInner = inputData->dStride;
        splitData.hOutputInner = inputData->hStride;
        splitData.wOutputInner = inputData->wStride;
        int64_t tmpCAligned = (inputData->cX < baseData.moveDataNumCacheLine) ? inputData->cX :
                                                                                baseData.moveDataNumCacheLine;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.dOutputInner = inputData->dX;
        splitData.hOutputInner = inputData->hX;
        splitData.wOutputInner = inputData->wX;
        splitData.cOutputInner = inputData->cX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);

    while (splitData.dOutputInner != 1 || splitData.hOutputInner != 1 ||
           splitData.wOutputInner > baseData.proDataNumInOneBeat) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDHW();
        } else {
            break;
        }
    }

    if (inputData->cX <= baseData.proDataNumInOneBeat) {
        splitData.cOutputInner = inputData->cX;
        return;
    } else if (IsMeetUBSize()) {
        splitData.cOutputInner = baseData.proDataNumInOneBeat;
        return;
    } else {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->cX / 2, baseData.proDataNumInOneBeat);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.cOutputInner = mid * baseData.proDataNumInOneBeat;
            if (IsMeetUBSize()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.cOutputInner = bestSplit * baseData.proDataNumInOneBeat;
    }
}

void Pool3DGradNDHWCSmallKernelCommonTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
    if (TrySplitN()) {
        return;
    }
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        if (TrySplitAlignD()) {
            return;
        }
        if (TrySplitAlignH()) {
            return;
        }
        if (TrySplitAlignW()) {
            return;
        }
        if (TrySplitAlignC()) {
            return;
        }
    }
    splitData.isCheckRange = 1;
    SplitUnalignDHWC();
}

void Pool3DGradNDHWCSmallKernelCommonTiling::DoUBTiling()
{
    SearchBestTiling();
    IsMeetUBSize();

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t wTail = inputData->wX % splitData.wOutputInner;
    splitData.wOutputTail = (wTail == 0) ? splitData.wOutputInner : wTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t hTail = inputData->hX % splitData.hOutputInner;
    splitData.hOutputTail = (hTail == 0) ? splitData.hOutputInner : hTail;

    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);
    int64_t dTail = inputData->dX % splitData.dOutputInner;
    splitData.dOutputTail = (dTail == 0) ? splitData.dOutputInner : dTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData->nX, splitData.nOutputInner);
    int64_t nTail = inputData->nX % splitData.nOutputInner;
    splitData.nOutputTail = (nTail == 0) ? splitData.nOutputInner : nTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData->cX, splitData.cOutputInner);
    int64_t cTail = inputData->cX % splitData.cOutputInner;
    splitData.cOutputTail = (cTail == 0) ? splitData.cOutputInner : cTail;
}

void Pool3DGradNDHWCSmallKernelCommonTiling::DoBlockTiling()
{
    // inner 变化后 outer/tail 需同步重算, 保证 SearchOuterSingle 调整过程中多核分配自洽
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t wTail = inputData->wX % splitData.wOutputInner;
    splitData.wOutputTail = (wTail == 0) ? splitData.wOutputInner : wTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t hTail = inputData->hX % splitData.hOutputInner;
    splitData.hOutputTail = (hTail == 0) ? splitData.hOutputInner : hTail;

    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);
    int64_t dTail = inputData->dX % splitData.dOutputInner;
    splitData.dOutputTail = (dTail == 0) ? splitData.dOutputInner : dTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData->nX, splitData.nOutputInner);
    int64_t nTail = inputData->nX % splitData.nOutputInner;
    splitData.nOutputTail = (nTail == 0) ? splitData.nOutputInner : nTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData->cX, splitData.cOutputInner);
    int64_t cTail = inputData->cX % splitData.cOutputInner;
    splitData.cOutputTail = (cTail == 0) ? splitData.cOutputInner : cTail;

    splitData.totalBaseBlockNum = splitData.nOutputOuter * splitData.cOutputOuter * splitData.dOutputOuter *
                                  splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void Pool3DGradNDHWCSmallKernelCommonTiling::SearchOuterSingle(int64_t& inner, int64_t step)
{
    int64_t lastNormCoreProcessNum = splitData.normalCoreProcessNum;
    while (inner > step) {
        inner -= step;
        DoBlockTiling();
        int64_t newNormCoreProcessNum = splitData.normalCoreProcessNum;
        if (newNormCoreProcessNum > lastNormCoreProcessNum) {
            inner += step;
            DoBlockTiling();
            break;
        }
        lastNormCoreProcessNum = newNormCoreProcessNum;
    }
}

void Pool3DGradNDHWCSmallKernelCommonTiling::SearchOuterSingleAligned(int64_t& inner, int64_t stride,
                                                                      int64_t kernelSize)
{
    if (kernelSize == 1) {
        // 窗口为单点, 任意 tile 边界均不切割窗口
        SearchOuterSingle(inner, 1);
        return;
    }
    if (inner <= stride) {
        return;
    }
    if (inner % stride != 0) {
        // 整轴(TrySplitN/对齐路径的未切分轴)未按 stride 对齐: 先向下对齐到 stride 网格;
        // CeilDiv(totalBaseBlockNum, totalCoreNum) 跳升则回退, 该轴不可切
        int64_t originInner = inner;
        int64_t lastNormCoreProcessNum = splitData.normalCoreProcessNum;
        inner = inner / stride * stride;
        DoBlockTiling();
        if (splitData.normalCoreProcessNum > lastNormCoreProcessNum) {
            inner = originInner;
            DoBlockTiling();
            return;
        }
    }
    SearchOuterSingle(inner, stride);
}

void Pool3DGradNDHWCSmallKernelCommonTiling::AdjustInnerSplitForMultiCore()
{
    DoBlockTiling();
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        return;
    }

    // N 轴(高轴): 各平面独立, 任意 inner 均安全
    SearchOuterSingle(splitData.nOutputInner, 1);
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        DoBufferCalculate();
        return;
    }

    if (splitData.isCheckRange == 1) {
        // 非对齐兜底路径: kernel 侧 IS_CHECK_RANGE=1 逐 lane 过滤越界写, 任意 inner 安全
        SearchOuterSingle(splitData.dOutputInner, 1);
        if (splitData.usedCoreNum >= baseData.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingle(splitData.hOutputInner, 1);
        if (splitData.usedCoreNum >= baseData.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingle(splitData.wOutputInner, 1);
    } else if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        // 对齐/N 路径(isCheckRange=0): kernel 侧 scatter 无越界过滤,
        // D/H/W tile 边界必须保持在 stride 网格上(无 pad 无 overlap 时窗口不跨边界)
        SearchOuterSingleAligned(splitData.dOutputInner, inputData->dStride, inputData->dKernel);
        if (splitData.usedCoreNum >= baseData.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingleAligned(splitData.hOutputInner, inputData->hStride, inputData->hKernel);
        if (splitData.usedCoreNum >= baseData.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingleAligned(splitData.wOutputInner, inputData->wStride, inputData->wKernel);
    }
    // isCheckRange=0 且有 pad/overlap 的 N 路径: D/H/W 为整轴(单 tile), 切分会产生跨界窗口,
    // 仅调整 N 轴
    DoBufferCalculate();
}

ge::graphStatus Pool3DGradNDHWCSmallKernelCommonTiling::DoOpTiling(gert::TilingContext* context)
{
    DoUBTiling();
    AdjustInnerSplitForMultiCore();
    DoBlockTiling(); // 安全冗余, 确保最终 outer/tail 与多核分配一致
    SetTilingData(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DGradNDHWCSmallKernelCommonTiling::PostTiling(gert::TilingContext* context, uint64_t key)
{
    context->SetTilingKey(key);
    context->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

Pool3DGradNDHWCSplitInfo& Pool3DGradNDHWCSmallKernelCommonTiling::GetSplitData() { return splitData; }

Pool3DGradNDHWCBaseInfo& Pool3DGradNDHWCSmallKernelCommonTiling::GetBaseData() { return baseData; }

} // namespace optiling
