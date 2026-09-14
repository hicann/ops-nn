
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_grad_with_argmax_simd_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "pool3d_grad_ncdhw_small_kernel_tiling.h"
#include "pool_grad_tiling_split_helper.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 5120;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t THRESHOLD = 2;
static constexpr int64_t KERNEL_OFFSET = 1;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

void Pool3DGradNCDHWSmallKernelCommonTiling::InitializationVars(gert::TilingContext* context_, int64_t ubSize_,
                                                                int64_t coreNum_)
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData->inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.indexBytes = inputData->isInt32Meet ? INT64_SIZE : INT32_SIZE;
    baseData.availableUb = ubSize_ - UB_RESVERVED_SIZE;
    baseData.totalCoreNum = coreNum_;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;
    int64_t oneBlockNumT2 = baseData.ubBlockSize / baseData.indexBytes;

    baseData.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);

    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT2;
    baseData.inputNCSize = inputData->nX * inputData->cX;

    baseData.isPad = 0;
    if (inputData->hPad != 0 || inputData->wPad != 0 || inputData->dPad != 0 || inputData->hPadBack != 0 ||
        inputData->wPadBack != 0 || inputData->dPadBack != 0) {
        baseData.isPad = 1;
    }
    baseData.dProBatchSize = PoolGradTiling::CalcProBatchSize(inputData->dKernel, inputData->dStride);
    baseData.hProBatchSize = PoolGradTiling::CalcProBatchSize(inputData->hKernel, inputData->hStride);
    baseData.wProBatchSize = PoolGradTiling::CalcProBatchSize(inputData->wKernel, inputData->wStride);

    baseData.isOverlap = 0;
    if (baseData.wProBatchSize != 1 || baseData.hProBatchSize != 1 || baseData.dProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}

void Pool3DGradNCDHWSmallKernelCommonTiling::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner + inputData->dKernel - 1, inputData->dStride);
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);
    int64_t wInputAligned = Ops::Base::CeilAlign(
        splitData.wOutputInner + ((inputData->wKernel - KERNEL_OFFSET) * inputData->wDilation) * DOUBLE,
        baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeDHW = dInputInner * hInputInner * wInputInnerAligned;
    int64_t outputPlaneSizeDHW = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;

    splitData.inputBufferSize = splitData.highAxisInner *
                                (splitData.dOutputInner +
                                 ((inputData->dKernel - KERNEL_OFFSET) * inputData->dDilation) * DOUBLE) *
                                (splitData.hOutputInner +
                                 ((inputData->hKernel - KERNEL_OFFSET) * inputData->hDilation) * DOUBLE) *
                                wInputAligned * baseData.inputBytes;
    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeDHW * baseData.inputBytes;
    splitData.argmaxBufferSize = splitData.highAxisInner * dInputInner * hInputInner * wInputInner *
                                 (inputData->isInt32Meet ? INT64_SIZE : INT32_SIZE);
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeDHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.inputBufferSize + splitData.outputBufferSize + splitData.gradBufferSize +
                                 splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
    if (baseData.isPad == 1) {
        splitData.totalBufferSize += splitData.inputBufferSize;
    }
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::IsMeetTargetCoreNum() const
{
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t tmpDOutputOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpDOutputOuter * tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >=
           baseData.coreUsedForBestPerformance;
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    if (baseData.inputBytes == FLOAT16_SIZE) {
        return splitData.totalBufferSize <= baseData.availableUb &&
               splitData.gradBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes;
    }
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::TrySplitNC()
{
    splitData.dOutputInner = inputData->dX;
    PoolGradTiling::PoolGradNchwDims dims{inputData->hX, inputData->wX, inputData->hStride, inputData->wStride};
    return PoolGradTiling::TrySplitNc(splitData, dims, baseData.inputNCSize, baseData.coreUsedForBestPerformance,
                                      *this);
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::TrySplitAlignD()
{
    return TrySplitAlignedAxis(&Pool3DGradNCDHWSplitInfo::dOutputInner, inputData->dX, inputData->dStride,
                               &Pool3DGradNCDHWSplitInfo::hOutputInner, inputData->hX,
                               &Pool3DGradNCDHWSplitInfo::wOutputInner, inputData->wX);
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::TrySplitAlignedAxis(int64_t Pool3DGradNCDHWSplitInfo::*axisInner,
                                                                 int64_t axisX, int64_t axisStride,
                                                                 int64_t Pool3DGradNCDHWSplitInfo::*otherInnerA,
                                                                 int64_t otherValA,
                                                                 int64_t Pool3DGradNCDHWSplitInfo::*otherInnerB,
                                                                 int64_t otherValB)
{
    splitData.highAxisInner = 1;
    splitData.*otherInnerA = otherValA;
    splitData.*otherInnerB = otherValB;
    splitData.*axisInner = axisStride;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        splitData.*axisInner = PoolGradTiling::SearchMaxSplit(*this, splitData, axisInner, axisStride,
                                                              Ops::Base::CeilDiv(axisX / 2, axisStride));
        return true;
    }
    return false;
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::TrySplitAlignH()
{
    return TrySplitAlignedAxis(&Pool3DGradNCDHWSplitInfo::hOutputInner, inputData->hX, inputData->hStride,
                               &Pool3DGradNCDHWSplitInfo::dOutputInner, inputData->dX,
                               &Pool3DGradNCDHWSplitInfo::wOutputInner, inputData->wX);
}

bool Pool3DGradNCDHWSmallKernelCommonTiling::TrySplitAlignW()
{
    return TrySplitAlignedAxis(&Pool3DGradNCDHWSplitInfo::wOutputInner, inputData->wX, inputData->wStride,
                               &Pool3DGradNCDHWSplitInfo::hOutputInner, inputData->hStride,
                               &Pool3DGradNCDHWSplitInfo::dOutputInner, inputData->dStride);
}

void Pool3DGradNCDHWSmallKernelCommonTiling::SplitUnalignDHW()
{
    splitData.highAxisInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData->hStride;
        splitData.wOutputInner = inputData->wStride;
        splitData.dOutputInner = inputData->dStride;
    } else {
        splitData.hOutputInner = inputData->hX;
        splitData.wOutputInner = inputData->wX;
        splitData.dOutputInner = inputData->dX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData->dX, splitData.dOutputInner);

    while (splitData.hOutputInner != 1 || splitData.dOutputInner != 1 ||
           splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDWH();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(inputData->wX, baseData.proDataNumInOneBeatT2);
    return;
}

void Pool3DGradNCDHWSmallKernelCommonTiling::DynamicAdjustmentDWH()
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

void Pool3DGradNCDHWSmallKernelCommonTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
    if (TrySplitNC()) {
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
    }
    splitData.isCheckRange = 1;
    SplitUnalignDHW();
    return;
}

void Pool3DGradNCDHWSmallKernelCommonTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    PoolGradTiling::CalcAxisOuterTail(inputData->wX, splitData.wOutputInner, splitData.wOutputOuter,
                                      splitData.wOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->hX, splitData.hOutputInner, splitData.hOutputOuter,
                                      splitData.hOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->dX, splitData.dOutputInner, splitData.dOutputOuter,
                                      splitData.dOutputTail);
    PoolGradTiling::CalcAxisOuterTail(baseData.inputNCSize, splitData.highAxisInner, splitData.highAxisOuter,
                                      splitData.highAxisTail);
}

void Pool3DGradNCDHWSmallKernelCommonTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter *
                                  splitData.dOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void Pool3DGradNCDHWSmallKernelCommonTiling::PrintBaseData() const
{
    OP_LOGD("MaxPool3DGradNCDHWSmallKernel", "[MaxPool3DGradNCDHWSmallKernel] PrintBaseData start running");

    std::ostringstream info;
    info << "baseData.vRegSize: " << baseData.vRegSize << std::endl;
    info << "baseData.ubBlockSize: " << baseData.ubBlockSize << std::endl;
    info << "baseData.inputBytes: " << baseData.inputBytes << std::endl;
    info << "baseData.indexBytes: " << baseData.indexBytes << std::endl;
    info << "baseData.availableUb: " << baseData.availableUb << std::endl;
    info << "baseData.maxDataNumInOneBlock: " << baseData.maxDataNumInOneBlock << std::endl;
    info << "baseData.proDataNumInOneBeatT2: " << baseData.proDataNumInOneBeatT2 << std::endl;
    info << "baseData.totalCoreNum: " << baseData.totalCoreNum << std::endl;
    info << "baseData.coreUsedForBestPerformance: " << baseData.coreUsedForBestPerformance << std::endl;
    info << "baseData.isPad: " << baseData.isPad << std::endl;
    info << "baseData.isOverlap: " << baseData.isOverlap << std::endl;
    info << "baseData.hProBatchSize: " << baseData.hProBatchSize << std::endl;
    info << "baseData.wProBatchSize: " << baseData.wProBatchSize << std::endl;
    info << "baseData.dProBatchSize: " << baseData.dProBatchSize << std::endl;
    info << "baseData.inputNCSize: " << baseData.inputNCSize << std::endl;

    OP_LOGI("MaxPool3DGradNCDHWSmallKernel", "%s", info.str().c_str());
}

void Pool3DGradNCDHWSmallKernelCommonTiling::PrintSplitData() const
{
    OP_LOGD("MaxPool3DGradNCDHWSmallKernel", "[MaxPool3DGradNCDHWSmallKernel] PrintSplitData start running");

    std::ostringstream info;
    info << "splitData.isCheckRange: " << splitData.isCheckRange << std::endl;

    info << "splitData.highAxisInner: " << splitData.highAxisInner << std::endl;
    info << "splitData.highAxisTail: " << splitData.highAxisTail << std::endl;
    info << "splitData.highAxisOuter: " << splitData.highAxisOuter << std::endl;

    info << "splitData.hOutputInner: " << splitData.hOutputInner << std::endl;
    info << "splitData.hOutputTail: " << splitData.hOutputTail << std::endl;
    info << "splitData.hOutputOuter: " << splitData.hOutputOuter << std::endl;

    info << "splitData.wOutputInner: " << splitData.wOutputInner << std::endl;
    info << "splitData.wOutputTail: " << splitData.wOutputTail << std::endl;
    info << "splitData.wOutputOuter: " << splitData.wOutputOuter << std::endl;

    info << "splitData.dOutputInner: " << splitData.dOutputInner << std::endl;
    info << "splitData.dOutputTail: " << splitData.dOutputTail << std::endl;
    info << "splitData.dOutputOuter: " << splitData.dOutputOuter << std::endl;

    info << "splitData.normalCoreProcessNum: " << splitData.normalCoreProcessNum << std::endl;
    info << "splitData.tailCoreProcessNum: " << splitData.tailCoreProcessNum << std::endl;
    info << "splitData.usedCoreNum: " << splitData.usedCoreNum << std::endl;
    info << "splitData.totalBaseBlockNum: " << splitData.totalBaseBlockNum << std::endl;

    info << "splitData.inputBufferSize: " << splitData.inputBufferSize << std::endl;
    info << "splitData.outputBufferSize: " << splitData.outputBufferSize << std::endl;
    info << "splitData.gradBufferSize: " << splitData.gradBufferSize << std::endl;
    info << "splitData.argmaxBufferSize: " << splitData.argmaxBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

    OP_LOGI("MaxPool3DGradNCDHWSmallKernel", "%s", info.str().c_str());
}

void Pool3DGradNCDHWSmallKernelCommonTiling::SetTilingData(gert::TilingContext* context)
{
    Pool3DGradNameSpace::Pool3DGradNCDHWTilingData*
        tilingData = context->GetTilingData<Pool3DGradNameSpace::Pool3DGradNCDHWTilingData>();
    tilingData->dArgmax = inputData->dGrad;
    tilingData->hArgmax = inputData->hGrad;
    tilingData->wArgmax = inputData->wGrad;
    tilingData->dOutput = inputData->dX;
    tilingData->hOutput = inputData->hX;
    tilingData->wOutput = inputData->wX;
    tilingData->dKernel = inputData->dKernel;
    tilingData->hKernel = inputData->hKernel;
    tilingData->wKernel = inputData->wKernel;
    tilingData->dStride = inputData->dStride;
    tilingData->hStride = inputData->hStride;
    tilingData->wStride = inputData->wStride;
    tilingData->padD = inputData->dPad;
    tilingData->padH = inputData->hPad;
    tilingData->padW = inputData->wPad;
    tilingData->padDBack = inputData->dPadBack;
    tilingData->padHBack = inputData->hPadBack;
    tilingData->padWBack = inputData->wPadBack;
    tilingData->dilationD = inputData->dDilation;
    tilingData->dilationH = inputData->hDilation;
    tilingData->dilationW = inputData->wDilation;
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
    tilingData->inputBufferSize = splitData.inputBufferSize;
    tilingData->outputBufferSize = splitData.outputBufferSize;
    tilingData->gradBufferSize = splitData.gradBufferSize;
    tilingData->argmaxBufferSize = splitData.argmaxBufferSize;
    tilingData->dProBatchSize = baseData.dProBatchSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
}

ge::graphStatus Pool3DGradNCDHWSmallKernelCommonTiling::DoOpTiling(gert::TilingContext* context)
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData(context);
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DGradNCDHWSmallKernelCommonTiling::PostTiling(gert::TilingContext* context_, uint64_t key)
{
    Pool3DGradNameSpace::Pool3DGradNCDHWTilingData*
        tilingData = context_->GetTilingData<Pool3DGradNameSpace::Pool3DGradNCDHWTilingData>();
    context_->SetTilingKey(key);
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

Pool3DGradNCDHWSplitInfo& Pool3DGradNCDHWSmallKernelCommonTiling::GetSplitData() { return splitData; }

Pool3DGradNCDHWBaseInfo& Pool3DGradNCDHWSmallKernelCommonTiling::GetBaseData() { return baseData; }

} // namespace optiling
