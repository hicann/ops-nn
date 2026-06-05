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
 * \file adaptive_avg_pool2d_grad_tiling.cpp
 * \brief
 *
 */

#include "adaptive_avg_pool2d_grad_big_kernel_tiling.h"

namespace optiling {
using namespace AdaptiveAvgPool2dGradOp;

void AdaptiveAvgPool2dGradTilingBigKernel::InitializationVars()
{
    gradInputN_ = inputData_.nGrad;
    gradInputC_ = inputData_.cGrad;
    gradInputH_ = inputData_.hGrad;
    gradInputW_ = inputData_.wGrad;

    gradOutputN_ = inputData_.nX;
    gradOutputC_ = inputData_.cX;
    gradOutputH_ = inputData_.hX;
    gradOutputW_ = inputData_.wX;

    baseData_.vRegSize = Ops::Base::GetVRegSize(context_);

    baseData_.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData_.inputBytes = inputData_.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE; 

    baseData_.availableUb = ubSize_ - UB_RESVERVED_SIZE - UB_TEMP_BUFF_SIZE; 
    baseData_.totalCoreNum = coreNum_; 
    baseData_.coreUsedForBestPerformance = baseData_.totalCoreNum; 

    int64_t oneBlockNum = baseData_.ubBlockSize / baseData_.inputBytes;

    baseData_.maxDataNumInOneBlock = oneBlockNum; 

    baseData_.proDataNumInOneBeatT2 = baseData_.vRegSize / baseData_.ubBlockSize * oneBlockNum;  
    baseData_.inputNCSize = gradOutputN_ * gradOutputC_;
}

void AdaptiveAvgPool2dGradTilingBigKernel::DoBufferCalculate()
{
    int64_t hInputInner = Ops::Base::CeilDiv(splitData_.hOutputInner * gradInputH_, gradOutputH_) + 1;
    int64_t wInputInner = Ops::Base::CeilDiv(splitData_.wOutputInner * gradInputW_, gradOutputW_) + 1; 

    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData_.wOutputInner, baseData_.maxDataNumInOneBlock);

    int64_t inputPlaneSizeDHW =  hInputInner * wInputInner;
    int64_t outputPlaneSizeDHW = splitData_.hOutputInner * wOutputInnerAligned;

    splitData_.gradInputBufferSize = Ops::Base::CeilAlign(splitData_.highAxisInner * inputPlaneSizeDHW * baseData_.inputBytes, baseData_.ubBlockSize);
    splitData_.outputBufferSize = splitData_.highAxisInner * outputPlaneSizeDHW * FLOAT32_SIZE; 

    int64_t tmpTotalBufferSize = splitData_.gradInputBufferSize + splitData_.outputBufferSize;
    splitData_.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool AdaptiveAvgPool2dGradTilingBigKernel::IsCapable()
{
    InitializationVars();
    kernelH_ = Ops::Base::CeilDiv(gradOutputH_, gradInputH_);
    kernelW_ = Ops::Base::CeilDiv(gradOutputW_, gradInputW_);
    if ((kernelH_ * kernelW_ < ADAPTIVE_BIG_KERNEL_SIZE) || (kernelW_ <= (baseData_.vRegSize / baseData_.inputBytes / THRESHOLD))) {
        return false;
    }

    splitData_.highAxisInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = std::min(gradOutputW_, baseData_.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

bool AdaptiveAvgPool2dGradTilingBigKernel::IsMeetTargetCoreNum()
{
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputInner); 
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData_.inputNCSize, splitData_.highAxisInner);

    return  tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= baseData_.coreUsedForBestPerformance;
}

bool AdaptiveAvgPool2dGradTilingBigKernel::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

bool AdaptiveAvgPool2dGradTilingBigKernel::TrySplitNC()
{
    splitData_.wOutputInner = gradOutputW_;
    splitData_.hOutputInner = gradOutputH_;
    splitData_.highAxisInner = Ops::Base::CeilDiv(baseData_.inputNCSize, baseData_.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData_.highAxisInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = baseData_.inputNCSize;
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData_.highAxisInner = mid;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData_.highAxisInner = bestSplit;
        return true;
    } else {
        return false;
    }
}

void AdaptiveAvgPool2dGradTilingBigKernel::DynamicAdjustmentAlignDWH()
{
    if (splitData_.hOutputInner > kernelH_) {
        splitData_.hOutputInner -= kernelH_;
        splitData_.hOutputOuter = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputInner);
        return;
    }
    if (splitData_.wOutputInner > kernelW_) {
        splitData_.wOutputInner -= kernelW_;
        splitData_.wOutputOuter = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputInner);
        return;
    }
}

void AdaptiveAvgPool2dGradTilingBigKernel::SplitAlignDHW()
{
    splitData_.highAxisInner = 1;

    splitData_.hOutputInner = gradOutputH_;
    splitData_.wOutputInner = gradOutputW_;

    splitData_.wOutputOuter = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputInner); 
    splitData_.hOutputOuter = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputInner);

    while (splitData_.hOutputInner > kernelH_ || splitData_.wOutputInner > baseData_.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentAlignDWH();
        } else {
            return;
        }
    }

    splitData_.wOutputInner = std::min(gradOutputW_, baseData_.proDataNumInOneBeatT2);
    return;
}

void AdaptiveAvgPool2dGradTilingBigKernel::DynamicAdjustmentDWH()
{
    if (splitData_.hOutputInner != 1) {
        splitData_.hOutputOuter++;
        splitData_.hOutputInner = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputOuter);
        return;
    }
    splitData_.wOutputOuter++;
    splitData_.wOutputInner = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputOuter);
}

void AdaptiveAvgPool2dGradTilingBigKernel::SplitUnalignDHW()
{
    splitData_.highAxisInner = 1;

    splitData_.hOutputInner = gradOutputH_;
    splitData_.wOutputInner = gradOutputW_;

    splitData_.wOutputOuter = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputInner); 
    splitData_.hOutputOuter = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputInner);

    while (splitData_.hOutputInner != 1 || splitData_.wOutputInner > baseData_.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDWH();
        } else {
            return;
        }
    }
    splitData_.wOutputInner = std::min(gradOutputW_, baseData_.proDataNumInOneBeatT2);
    return;
}

void AdaptiveAvgPool2dGradTilingBigKernel::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }
    SplitUnalignDHW();
    return;
}

void AdaptiveAvgPool2dGradTilingBigKernel::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData_.wOutputOuter = Ops::Base::CeilDiv(gradOutputW_, splitData_.wOutputInner); 
    int64_t tempWOutputTail = gradOutputW_ % splitData_.wOutputInner; 
    splitData_.wOutputTail = tempWOutputTail == 0 ? splitData_.wOutputInner : tempWOutputTail; 

    splitData_.hOutputOuter = Ops::Base::CeilDiv(gradOutputH_, splitData_.hOutputInner);
    int64_t tempHOutputTail = gradOutputH_ % splitData_.hOutputInner;
    splitData_.hOutputTail = tempHOutputTail == 0 ? splitData_.hOutputInner : tempHOutputTail;

    splitData_.highAxisOuter = Ops::Base::CeilDiv(baseData_.inputNCSize, splitData_.highAxisInner);
    int64_t tempHighAxisTail = baseData_.inputNCSize % splitData_.highAxisInner;
    splitData_.highAxisTail = tempHighAxisTail == 0 ? splitData_.highAxisInner : tempHighAxisTail;
}

void AdaptiveAvgPool2dGradTilingBigKernel::DoBlockTiling()
{
    splitData_.totalBaseBlockNum = splitData_.highAxisOuter * splitData_.hOutputOuter * splitData_.wOutputOuter; 
    splitData_.normalCoreProcessNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, baseData_.totalCoreNum); 
    splitData_.usedCoreNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, splitData_.normalCoreProcessNum); 
    splitData_.tailCoreProcessNum =
        splitData_.totalBaseBlockNum - splitData_.normalCoreProcessNum * (splitData_.usedCoreNum - 1); 
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBigKernel::SetTilingData()
{
    AdaptiveAvgPool2dGradOp::AdaptiveAvgPool2dGradBigKernelTiling* tilingData =
        context_->GetTilingData<AdaptiveAvgPool2dGradOp::AdaptiveAvgPool2dGradBigKernelTiling>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->hInput = gradInputH_;
    tilingData->wInput = gradInputW_;
    tilingData->hOutput = gradOutputH_;
    tilingData->wOutput = gradOutputW_;
    tilingData->highAxisInner = splitData_.highAxisInner;
    tilingData->highAxisTail = splitData_.highAxisTail;
    tilingData->highAxisOuter = splitData_.highAxisOuter;
    tilingData->hOutputInner = splitData_.hOutputInner;
    tilingData->hOutputTail = splitData_.hOutputTail;
    tilingData->hOutputOuter = splitData_.hOutputOuter;
    tilingData->wOutputInner = splitData_.wOutputInner;
    tilingData->wOutputTail = splitData_.wOutputTail;
    tilingData->wOutputOuter = splitData_.wOutputOuter;
    tilingData->normalCoreProcessNum = splitData_.normalCoreProcessNum;
    tilingData->tailCoreProcessNum = splitData_.tailCoreProcessNum;
    tilingData->usedCoreNum = splitData_.usedCoreNum;
    tilingData->outputBufferSize = splitData_.outputBufferSize;
    tilingData->gradInputBufferSize = splitData_.gradInputBufferSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBigKernel::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    return SetTilingData();
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBigKernel::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool2dGradTilingBigKernel::GetTilingKey() const
{
    int64_t outDataCount = inputData_.nX * inputData_.cX * inputData_.hX * inputData_.wX;
    bool needInt64 = (outDataCount > static_cast<int64_t>(INT32_MAX) ||
                      inputData_.hGrad * inputData_.hX > static_cast<int64_t>(INT32_MAX) ||
                      inputData_.wGrad * inputData_.wX > static_cast<int64_t>(INT32_MAX));
    uint32_t idxDtype = needInt64 ? TPL_INT64 : TPL_INT32;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_BIG_KERNEL, idxDtype, isChannelLast);
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBigKernel::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(splitData_.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dGradTilingBigKernel::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2dGrad, AdaptiveAvgPool2dGradTilingBigKernel, 10);
} // namespace optiling
