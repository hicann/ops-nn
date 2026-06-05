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
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoBufferCalculate()
{
    splitData.inputQueBufferSize = 0;
    splitData.transQueBufferSize = 0;
    splitData.transOutQueBufferSize = 0;
    splitData.totalBufferSize = 0;

    const int64_t transRowAlign = TRANS_ADDR_LEN;
    const int64_t transColAlign = baseData.maxDataNumInOneBlock;

    const int64_t hInputInner =
        Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    const int64_t wInputInner =
        Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1;

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

    splitData.totalBufferSize =
        BUFFER_NUM * (splitData.inputQueBufferSize +
                      splitData.transQueBufferSize +
                      splitData.transOutQueBufferSize);
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::IsCapable()
{
    InitializationVars();

    kernelH = Ops::Base::CeilDiv(gradOutputH, gradInputH);
    kernelW = Ops::Base::CeilDiv(gradOutputW, gradInputW);

    if (kernelH * kernelW >= KERNEL_SIZE_MAX ||
        baseData.inputNCSize < HIGH_THRESHOLD ||
        gradInputW * gradInputH < WINSIZE_THRESHOLD) {
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

    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.hOutputInner = LIMIT;
    splitData.wOutputInner = LIMIT;
    DoBufferCalculate();

    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool2dGradTilingSmallKernel::IsMeetTargetCoreNum()
{
    const int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    const int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    const int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >=
           baseData.coreUsedForBestPerformance;
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
    if (splitData.hOutputInner != LIMIT) {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputOuter);
        return;
    }

    if (splitData.wOutputInner != LIMIT) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputOuter);
        return;
    }
}

void AdaptiveAvgPool2dGradTilingSmallKernel::SplitUnalignHW()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;

    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);

    while (splitData.hOutputInner != LIMIT || splitData.wOutputInner != LIMIT) {
        if (IsMeetTargetCoreNum() && IsMeetUBSize()) {
            return;
        }
        DynamicAdjustmentHW();
    }

    DoBufferCalculate();
}

void AdaptiveAvgPool2dGradTilingSmallKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }

    SplitUnalignHW();
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
    splitData.wOutputTail =
        (gradOutputW % splitData.wOutputInner == 0) ? splitData.wOutputInner : (gradOutputW % splitData.wOutputInner);

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.hOutputTail =
        (gradOutputH % splitData.hOutputInner == 0) ? splitData.hOutputInner : (gradOutputH % splitData.hOutputInner);

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    splitData.highAxisTail =
        (baseData.inputNCSize % splitData.highAxisInner == 0) ?
            splitData.highAxisInner :
            (baseData.inputNCSize % splitData.highAxisInner);
}

void AdaptiveAvgPool2dGradTilingSmallKernel::DoBlockTiling()
{
    splitData.totalBaseBlockNum =
        splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter;

    splitData.normalCoreProcessNum =
        Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum =
        Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
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
    OP_LOGD("AdaptiveAvgPool2dGradNCHW", "[AdaptiveAvgPool2dGradNCHW] PrintSplitData start running");

    std::ostringstream info;
    info << "baseData.availableUb: " << baseData.availableUb << std::endl;

    info << "splitData.highAxisInner: " << splitData.highAxisInner << std::endl;
    info << "splitData.highAxisTail: " << splitData.highAxisTail << std::endl;
    info << "splitData.highAxisOuter: " << splitData.highAxisOuter << std::endl;

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

    OP_LOGI("AdaptiveAvgPool2dGradNCHW", "%s", info.str().c_str());
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
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingSmallKernel::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradTilingSmallKernel, 20);
} // namespace optiling