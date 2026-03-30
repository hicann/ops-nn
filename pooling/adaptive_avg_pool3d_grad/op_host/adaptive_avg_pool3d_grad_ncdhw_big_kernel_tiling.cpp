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

#include "adaptive_avg_pool3d_grad_ncdhw_big_kernel_tiling.h"

namespace optiling {
using namespace AdaptiveAvgPool3dGradOp;

void AdaptiveAvgPool3dGradTilingBigKernel::InitializationVars()
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

    int64_t oneBlockNum = baseData.ubBlockSize / baseData.inputBytes;

    baseData.maxDataNumInOneBlock = oneBlockNum; 

    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * oneBlockNum;  
    baseData.inputNCSize = gradOutputN * gradOutputC;
}

void AdaptiveAvgPool3dGradTilingBigKernel::DoBufferCalculate()
{
    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner * gradInputD, gradOutputD) + 1;
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1; 

    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeDHW = dInputInner * hInputInner * wInputInner;
    int64_t outputPlaneSizeDHW = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;

    splitData.gradInputBufferSize = Ops::Base::CeilAlign(splitData.highAxisInner * inputPlaneSizeDHW * baseData.inputBytes, ALIGN_NUM);
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeDHW * FLOAT32_SIZE; 

    int64_t tmpTotalBufferSize = splitData.gradInputBufferSize + splitData.outputBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::IsCapable()
{
    InitializationVars();
    if(inputData.inputFormat != ge::Format::FORMAT_NCDHW)
    {
        return false;
    }
    kernelD = Ops::Base::CeilDiv(gradOutputD, gradInputD);
    kernelH = Ops::Base::CeilDiv(gradOutputH, gradInputH);
    kernelW = Ops::Base::CeilDiv(gradOutputW, gradInputW);
    if ((kernelD * kernelH * kernelW < ADAPTIVE_BIG_KERNEL_SIZE) || (kernelW <= (baseData.vRegSize / baseData.inputBytes / 2))) {
        return false;
    }

    splitData.highAxisInner = 1;
    splitData.dOutputInner = 1;
    splitData.hOutputInner = 1;

    splitData.wOutputInner = std::min(gradOutputW, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::IsMeetTargetCoreNum()
{
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner); 
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    int64_t tmpDOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpDOutputOuter * tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::TrySplitNC()
{
    splitData.wOutputInner = gradOutputW;
    splitData.hOutputInner = gradOutputH;
    splitData.dOutputInner = gradOutputD;
    splitData.highAxisInner = Ops::Base::CeilDiv(baseData.inputNCSize, baseData.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.highAxisInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = baseData.inputNCSize;
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.highAxisInner = mid;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.highAxisInner = bestSplit;
        return true;
    } else {
        return false;
    }
}

void AdaptiveAvgPool3dGradTilingBigKernel::DynamicAdjustmentAlignDWH()
{
    if (splitData.dOutputInner > kernelD) {
        splitData.dOutputInner -= kernelD;
        splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
        return;
    } 
    if (splitData.hOutputInner > kernelH) {
        splitData.hOutputInner -= kernelH;
        splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
        return;
    }
    if (splitData.wOutputInner > kernelW) {
        splitData.wOutputInner -= kernelW;
        splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
        return;
    }
}

void AdaptiveAvgPool3dGradTilingBigKernel::SplitAlignDHW()
{
    splitData.highAxisInner = 1;

    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;
    splitData.dOutputInner = gradOutputD;

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner); 
    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);

    while (splitData.dOutputInner > kernelD || splitData.hOutputInner > kernelH || splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentAlignDWH();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(gradOutputW, baseData.proDataNumInOneBeatT2);
    return;
}

void AdaptiveAvgPool3dGradTilingBigKernel::DynamicAdjustmentDWH()
{
    if (splitData.dOutputInner != 1) {
        splitData.dOutputOuter++;
        splitData.dOutputInner = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputOuter);
        return;
    } 
    if (splitData.hOutputInner != 1) {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputOuter);
        return;
    }
    splitData.wOutputOuter++;
    splitData.wOutputInner = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputOuter);
}

void AdaptiveAvgPool3dGradTilingBigKernel::SplitUnalignDHW()
{
    splitData.highAxisInner = 1;

    splitData.hOutputInner = gradOutputH;
    splitData.wOutputInner = gradOutputW;
    splitData.dOutputInner = gradOutputD;

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner); 
    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);

    while (splitData.hOutputInner != 1 || splitData.dOutputInner != 1 || splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDWH();
        } else {
            return;
        }
    }
    splitData.wOutputInner = std::min(gradOutputW, baseData.proDataNumInOneBeatT2);
    return;
}

void AdaptiveAvgPool3dGradTilingBigKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }
    SplitUnalignDHW();
    return;
}

void AdaptiveAvgPool3dGradTilingBigKernel::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData.wOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner); 
    int64_t tempWOutputTail = gradOutputW % splitData.wOutputInner; 
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail; 

    splitData.hOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
    int64_t tempHOutputTail = gradOutputH % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.dOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
    int64_t tempDOutputTail = gradOutputD % splitData.dOutputInner;
    splitData.dOutputTail = tempDOutputTail == 0 ? splitData.dOutputInner : tempDOutputTail;

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    int64_t tempHighAxisTail = baseData.inputNCSize % splitData.highAxisInner;
    splitData.highAxisTail = tempHighAxisTail == 0 ? splitData.highAxisInner : tempHighAxisTail;
}

void AdaptiveAvgPool3dGradTilingBigKernel::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter * splitData.dOutputOuter; 
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum); 
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum); 
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1); 
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::SetTilingData()
{
    AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35* tilingData =
        context_->GetTilingData<AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

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
    tilingData->outputBufferSize = splitData.outputBufferSize;
    tilingData->gradInputBufferSize = splitData.gradInputBufferSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    return SetTilingData();
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool3dGradTilingBigKernel::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    uint32_t idxDtype = outDataCount <= static_cast<int64_t>(MAX_INT32) ? TPL_INT32 : TPL_INT64;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_BIG_KERNEL, idxDtype, isChannelLast);
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingBigKernel, 10);
} // namespace optiling
