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

namespace optiling {
using namespace AdaptiveAvgPool3dGradOp;

constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t KERNEL_SIZE_MAX = 256;
constexpr uint64_t HIGH_THRESHOLD = 128;
constexpr uint64_t WINSIZE_THRESHOLD = 16;
constexpr uint64_t INPUTW_FLOAT_THRESHOLD = 8;
constexpr uint64_t INPUTW_BFLOAT_THRESHOLD = 16;
constexpr uint64_t LIMIT = 1;

//保证 nextInner 一定严格变小，避免 while 卡死。
static inline int64_t ShrinkInnerStrict(int64_t total, int64_t curInner)
{
    if (curInner <= LIMIT) {
        return LIMIT;
    }

    const int64_t curOuter = Ops::Base::CeilDiv(total, curInner);
    int64_t nextInner = Ops::Base::CeilDiv(total, curOuter + 1);

    // 反推结果如果没有变小，就强制减 1，保证循环一定前进
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
    splitData.computeSrcBufferSize = 0;
    splitData.computeAccumBufferSize = 0;
    splitData.totalBufferSize = 0;

    const int64_t transRowAlign = TRANS_ADDR_LEN;
    const int64_t transColAlign = baseData.maxDataNumInOneBlock;
    const bool needFp32Scratch = (inputData.inputDtype != ge::DT_FLOAT);

    const int64_t dInputInner =
        Ops::Base::CeilDiv(splitData.dOutputInner * gradInputD, gradOutputD) + 1;
    const int64_t hInputInner =
        Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    const int64_t wInputInner =
        Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1;

    const int64_t highAxisInner = splitData.highAxisInner;
    const int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, transColAlign);
    const int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, transColAlign);

    const int64_t inputColNum = dInputInner * hInputInner * wInputInnerAligned;
    const int64_t inputBytes = highAxisInner * inputColNum * baseData.inputBytes;

    const int64_t outputRowNum = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;
    const int64_t outputRowNumAligned = Ops::Base::CeilAlign(outputRowNum, transRowAlign);
    const int64_t outputBytes = outputRowNumAligned * highAxisInner * baseData.inputBytes;

    splitData.inputQueBufferSize = Ops::Base::CeilAlign(inputBytes, baseData.ubBlockSize);
    splitData.transQueBufferSize = Ops::Base::CeilAlign(std::max(inputBytes, outputBytes), baseData.ubBlockSize);
    splitData.transOutQueBufferSize = Ops::Base::CeilAlign(outputBytes, baseData.ubBlockSize);

    if (needFp32Scratch) {
        splitData.computeSrcBufferSize = Ops::Base::CeilAlign(
            (splitData.transQueBufferSize / baseData.inputBytes) * FLOAT32_SIZE,
            baseData.ubBlockSize);

        splitData.computeAccumBufferSize = Ops::Base::CeilAlign(
            (splitData.transOutQueBufferSize / baseData.inputBytes) * FLOAT32_SIZE,
            baseData.ubBlockSize);
    }

    splitData.totalBufferSize =
        BUFFER_NUM * (splitData.inputQueBufferSize +
                      splitData.transQueBufferSize +
                      splitData.transOutQueBufferSize) +
        splitData.computeSrcBufferSize +
        splitData.computeAccumBufferSize;
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

    if (kernelD * kernelH * kernelW >= KERNEL_SIZE_MAX ||
        baseData.inputNCSize < HIGH_THRESHOLD ||
        gradInputW * gradInputH * gradInputD < WINSIZE_THRESHOLD) {
        return false;
    }

    if(inputData.inputDtype == ge::DT_FLOAT) {
        if(gradInputW < INPUTW_FLOAT_THRESHOLD) {
            return false;
        }
    } else {
        if(gradInputW < INPUTW_BFLOAT_THRESHOLD) {
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
    // 固定DHW面，只切NC -- NC长度固定64或者128
    splitData.wOutputInner = gradOutputW;
    splitData.hOutputInner = gradOutputH;
    splitData.dOutputInner = gradOutputD;
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;

    return IsMeetUBSize() && IsMeetTargetCoreNum();
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DynamicAdjustmentAlignDWH()
{
    if (splitData.dOutputInner > kernelD) {
        splitData.dOutputInner -= kernelD;
        return;
    }
    if (splitData.hOutputInner > kernelH) {
        splitData.hOutputInner -= kernelH;
        return;
    }
    if (splitData.wOutputInner > kernelW) {
        splitData.wOutputInner -= kernelW;
        return;
    }
}

void AdaptiveAvgPool3dGradTilingSmallKernel::SplitAlignDHW()
{
    splitData.highAxisInner = baseData.proDataNumInOneBeatT2;
    splitData.dOutputInner = gradOutputD;
    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;

    while (splitData.dOutputInner > kernelD ||
           splitData.hOutputInner > kernelH ||
           splitData.wOutputInner > kernelW) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentAlignDWH();
        } else {
            return;
        }
    }
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

        if (oldD == splitData.dOutputInner &&
            oldH == splitData.hOutputInner &&
            oldW == splitData.wOutputInner) {
            break;
        }
    }
}

void AdaptiveAvgPool3dGradTilingSmallKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }

    SplitAlignDHW();
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
    splitData.wOutputTail =
        (gradOutputW % splitData.wOutputInner == 0) ? splitData.wOutputInner :
                                                      (gradOutputW % splitData.wOutputInner);

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.hOutputTail =
        (gradOutputH % splitData.hOutputInner == 0) ? splitData.hOutputInner :
                                                      (gradOutputH % splitData.hOutputInner);

    splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
    splitData.dOutputTail =
        (gradOutputD % splitData.dOutputInner == 0) ? splitData.dOutputInner :
                                                      (gradOutputD % splitData.dOutputInner);

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    splitData.highAxisTail =
        (baseData.inputNCSize % splitData.highAxisInner == 0) ? splitData.highAxisInner :
                                                                (baseData.inputNCSize % splitData.highAxisInner);
}

void AdaptiveAvgPool3dGradTilingSmallKernel::DoBlockTiling()
{
    splitData.totalBaseBlockNum =
        splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter * splitData.dOutputOuter;

    splitData.normalCoreProcessNum =
        Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum =
        Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
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

    std::ostringstream info;
    info << "baseData.availableUb: " << baseData.availableUb << std::endl;

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
    info << "splitData.computeSrcBufferSize: " << splitData.computeSrcBufferSize << std::endl;
    info << "splitData.computeAccumBufferSize: " << splitData.computeAccumBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

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

ge::graphStatus AdaptiveAvgPool3dGradTilingSmallKernel::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingSmallKernel, 20);
} // namespace optiling
