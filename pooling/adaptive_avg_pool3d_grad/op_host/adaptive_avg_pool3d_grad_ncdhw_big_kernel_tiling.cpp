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
    InitCommonVars();
    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * baseData.maxDataNumInOneBlock;
}

void AdaptiveAvgPool3dGradTilingBigKernel::DoBufferCalculate()
{
    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner * gradInputD, gradOutputD) + 1;
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner * gradInputH, gradOutputH) + 1;
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner * gradInputW, gradOutputW) + 1;

    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeDHW = dInputInner * hInputInner * wInputInner;
    int64_t outputPlaneSizeDHW = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;

    splitData.gradInputBufferSize = Ops::Base::CeilAlign(
        splitData.highAxisInner * inputPlaneSizeDHW * baseData.inputBytes, ALIGN_NUM);
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeDHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.gradInputBufferSize + splitData.outputBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::IsCapable()
{
    InitializationVars();
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }
    kernelD = Ops::Base::CeilDiv(gradOutputD, gradInputD);
    kernelH = Ops::Base::CeilDiv(gradOutputH, gradInputH);
    kernelW = Ops::Base::CeilDiv(gradOutputW, gradInputW);
    if ((kernelD * kernelH * kernelW < ADAPTIVE_BIG_KERNEL_SIZE) ||
        (kernelW <= (baseData.vRegSize / baseData.inputBytes / DOUBLE_BUFFER))) {
        return false;
    }

    splitData.highAxisInner = 1;
    splitData.dOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = std::min(gradOutputW, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AdaptiveAvgPool3dGradTilingBigKernel::TrySplitNC()
{
    splitData.dOutputInner = gradOutputD;
    PoolGradTiling::PoolGradNchwDims dims{gradOutputH, gradOutputW, 0, 0};
    return PoolGradTiling::TrySplitNc(splitData, dims, baseData.inputNCSize, baseData.coreUsedForBestPerformance,
                                      *this);
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

    while (splitData.dOutputInner > kernelD || splitData.hOutputInner > kernelH ||
           splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
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

    while (splitData.hOutputInner != 1 || splitData.dOutputInner != 1 ||
           splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
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

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::SetTilingData()
{
    AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35* tilingData = context_->GetTilingData<
        AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    SetCommonTilingData(tilingData);
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

uint64_t AdaptiveAvgPool3dGradTilingBigKernel::GetTilingKey() const
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    int64_t inDataCount = inputData.nGrad * inputData.cGrad * inputData.dGrad * inputData.hGrad * inputData.wGrad;
    bool needInt64 = (outDataCount > static_cast<int64_t>(MAX_INT32) || inDataCount > static_cast<int64_t>(MAX_INT32) ||
                      inputData.dGrad * inputData.dX > static_cast<int64_t>(MAX_INT32) ||
                      inputData.hGrad * inputData.hX > static_cast<int64_t>(MAX_INT32) ||
                      inputData.wGrad * inputData.wX > static_cast<int64_t>(MAX_INT32));
    uint32_t idxDtype = needInt64 ? TPL_INT64 : TPL_INT32;
    uint32_t isChannelLast = 0;
    return GET_TPL_TILING_KEY(TPL_BIG_KERNEL, idxDtype, isChannelLast);
}

ge::graphStatus AdaptiveAvgPool3dGradTilingBigKernel::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingBigKernel, 10);
} // namespace optiling
