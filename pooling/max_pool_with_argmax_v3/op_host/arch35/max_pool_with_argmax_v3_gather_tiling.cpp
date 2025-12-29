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
 * \file max_pool_with_argmax_v3_gather_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_with_argmax_v3_gather_tiling.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 0;
static constexpr int64_t HELPER_BUFFER_SIZE = 1024;
static constexpr int64_t NO_PADDING_TILING_KEY = 300001;
static constexpr int64_t PADDING_TILING_KEY = 300002;
static constexpr int64_t MAX_BANDWIDTH_COEFFICIENTS = 2;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MIN_OUTPUT_THRESHOLD = 32;
static constexpr int64_t DILATION_THRESHOLD = 1;

void MaxPoolWithArgmaxV3GatherTiling::InitializationVars()
{
    baseData_.inputBytes = dtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData_.indexBytes = inputData.indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData_.availableUb = ubSize - UB_RESVERVED_SIZE;
    baseData_.totalCoreNum = coreNum;
    baseData_.coreUsedForBestPerformance = baseData_.inputBytes == FLOAT32_SIZE ?
                                               baseData_.totalCoreNum / MAX_BANDWIDTH_COEFFICIENTS :
                                               baseData_.totalCoreNum;

    baseData_.padTop = inputData.pad[H_DIM];
    baseData_.padLeft = inputData.pad[W_DIM];
    baseData_.hInput = inputData.inputShape[H_DIM];
    baseData_.wInput = inputData.inputShape[W_DIM];
    baseData_.hOutput = inputData.outShape[H_DIM];
    baseData_.wOutput = inputData.outShape[W_DIM];
    baseData_.hStride = inputData.stride[H_DIM];
    baseData_.wStride = inputData.stride[W_DIM];
    baseData_.hKernel = inputData.kernelSize[H_DIM];
    baseData_.wKernel = inputData.kernelSize[W_DIM];
    baseData_.highAxisTotal = inputData.batches;
    baseData_.hDilation = inputData.dilation[H_DIM];
    baseData_.wDilation = inputData.dilation[W_DIM];
    baseData_.isPad = 0;
    if (baseData_.padTop != 0 || baseData_.padLeft != 0) {
        baseData_.isPad = 1;
    }

    if (inputData.ceilMode && baseData_.isPad == 0) {
        if (((baseData_.wOutput - 1) * baseData_.wStride + baseData_.wKernel) != baseData_.wInput ||
            ((baseData_.hOutput - 1) * baseData_.hStride + baseData_.hKernel) != baseData_.hInput) {
            baseData_.isPad = 1;
        }
    }

    baseData_.oneBlockNumT1 = Ops::Base::GetUbBlockSize(context_) / baseData_.inputBytes;
    baseData_.oneBlockNumT2 = Ops::Base::GetUbBlockSize(context_) / baseData_.indexBytes;
}

bool MaxPoolWithArgmaxV3GatherTiling::IsCapable()
{
    if (inputData.dilation[H_DIM] > DILATION_THRESHOLD || inputData.dilation[W_DIM] > DILATION_THRESHOLD ||
        inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    InitializationVars();
    if (baseData_.wKernel * baseData_.inputBytes >= CACHE_LINE_SIZE) {
        return false;
    }

    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;
    splitData_.highAxisInner = 1;
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb / MIN_OUTPUT_THRESHOLD;
}

uint64_t MaxPoolWithArgmaxV3GatherTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_PADDING_TILING_KEY;
    if (baseData_.isPad == 1) {
        tilingKey = PADDING_TILING_KEY;
    }
    return tilingKey;
}

void MaxPoolWithArgmaxV3GatherTiling::DoBufferCalculate()
{
    splitData_.hInputInner =
        (splitData_.hOutputInner - 1) * baseData_.hStride + (baseData_.hKernel - 1) * baseData_.hDilation + 1;
    splitData_.wInputInner =
        (splitData_.wOutputInner - 1) * baseData_.wStride + (baseData_.wKernel - 1) * baseData_.wDilation + 1;
    int64_t maxDataNumInOneBlock = std::max(baseData_.oneBlockNumT1, baseData_.oneBlockNumT2);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(splitData_.wInputInner, baseData_.oneBlockNumT1);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData_.wOutputInner, maxDataNumInOneBlock);

    int64_t inputBufferSize =
        splitData_.highAxisInner * splitData_.hInputInner * wInputInnerAligned * baseData_.inputBytes;
    splitData_.inputBufferSize = inputBufferSize;
    // pad情况下COPY IN的UB地址不一定32字节对齐
    if (baseData_.isPad == 1) {
        inputBufferSize *= DOUBLE;
    }
    int64_t outputDataSize = splitData_.highAxisInner * splitData_.hOutputInner * wOutputInnerAligned;
    splitData_.maxValueBufferSize = outputDataSize * baseData_.inputBytes;
    splitData_.argmaxBufferSize = outputDataSize * baseData_.indexBytes;

    int64_t tmpTotalBufferSize =
        inputBufferSize + splitData_.maxValueBufferSize + splitData_.argmaxBufferSize + HELPER_BUFFER_SIZE;

    splitData_.totalBufferSize = tmpTotalBufferSize * DOUBLE;
    if (baseData_.isPad == 1) {
        splitData_.totalBufferSize -= splitData_.inputBufferSize;
    }
}

bool MaxPoolWithArgmaxV3GatherTiling::IsMeetTargetCoreNum() const
{
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.highAxisInner);
    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter >= baseData_.coreUsedForBestPerformance;
}

bool MaxPoolWithArgmaxV3GatherTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

void MaxPoolWithArgmaxV3GatherTiling::BinarySearch(int64_t start, int64_t end, int64_t* value)
{
    int64_t left = start;
    int64_t right = end;
    int64_t bestSplit = 1;

    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        *value = mid;
        if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    *value = bestSplit;
}

bool MaxPoolWithArgmaxV3GatherTiling::TrySplitNC()
{
    splitData_.hOutputInner = baseData_.hOutput;
    splitData_.wOutputInner = baseData_.wOutput;

    splitData_.highAxisInner = Ops::Base::CeilDiv(baseData_.highAxisTotal, baseData_.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData_.highAxisInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.highAxisTotal, &splitData_.highAxisInner);
        return true;
    } else {
        return false;
    }
}

bool MaxPoolWithArgmaxV3GatherTiling::TrySplitH()
{
    splitData_.highAxisInner = 1;
    splitData_.wOutputInner = baseData_.wOutput;
    splitData_.hOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.hOutput, &splitData_.hOutputInner);
        return true;
    } else {
        return false;
    }
}

bool MaxPoolWithArgmaxV3GatherTiling::TrySplitW()
{
    splitData_.highAxisInner = 1;
    splitData_.hOutputInner = 1;

    splitData_.wOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.wOutput, &splitData_.wOutputInner);
        return true;
    } else {
        return false;
    }
}

void MaxPoolWithArgmaxV3GatherTiling::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }
    if (TrySplitH()) {
        return;
    }
    if (TrySplitW()) {
        return;
    }
}

void MaxPoolWithArgmaxV3GatherTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData_.wOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tempWOutputTail = baseData_.wOutput % splitData_.wOutputInner;
    splitData_.wOutputTail = tempWOutputTail == 0 ? splitData_.wOutputInner : tempWOutputTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tempHOutputTail = baseData_.hOutput % splitData_.hOutputInner;
    splitData_.hOutputTail = tempHOutputTail == 0 ? splitData_.hOutputInner : tempHOutputTail;

    splitData_.highAxisOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.highAxisInner);
    int64_t tempNOutputTail = baseData_.highAxisTotal % splitData_.highAxisInner;
    splitData_.highAxisTail = tempNOutputTail == 0 ? splitData_.highAxisInner : tempNOutputTail;
}

void MaxPoolWithArgmaxV3GatherTiling::DoBlockTiling()
{
    splitData_.totalBaseBlockNum = splitData_.highAxisOuter * splitData_.hOutputOuter * splitData_.wOutputOuter;
    splitData_.normalCoreProcessNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, baseData_.totalCoreNum);
    splitData_.usedCoreNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, splitData_.normalCoreProcessNum);
    splitData_.tailCoreProcessNum =
        splitData_.totalBaseBlockNum - splitData_.normalCoreProcessNum * (splitData_.usedCoreNum - 1);
}

void MaxPoolWithArgmaxV3GatherTiling::PrintBaseData() const
{
    OP_LOGI("PrintBaseData", "%s", baseData_.ToString().c_str());
}

void MaxPoolWithArgmaxV3GatherTiling::PrintSplitData() const
{
    OP_LOGI("PrintSplitData", "%s", splitData_.ToString().c_str());
}

void MaxPoolWithArgmaxV3GatherTiling::SetTilingData()
{
    tilingData_.set_hInput(baseData_.hInput);
    tilingData_.set_wInput(baseData_.wInput);
    tilingData_.set_hOutput(baseData_.hOutput);
    tilingData_.set_wOutput(baseData_.wOutput);
    tilingData_.set_hKernel(baseData_.hKernel);
    tilingData_.set_wKernel(baseData_.wKernel);
    tilingData_.set_hStride(baseData_.hStride);
    tilingData_.set_wStride(baseData_.wStride);
    tilingData_.set_padTop(baseData_.padTop);
    tilingData_.set_padLeft(baseData_.padLeft);
    tilingData_.set_highAxisInner(splitData_.highAxisInner);
    tilingData_.set_highAxisTail(splitData_.highAxisTail);
    tilingData_.set_highAxisOuter(splitData_.highAxisOuter);
    tilingData_.set_hOutputInner(splitData_.hOutputInner);
    tilingData_.set_hOutputTail(splitData_.hOutputTail);
    tilingData_.set_hOutputOuter(splitData_.hOutputOuter);
    tilingData_.set_wOutputInner(splitData_.wOutputInner);
    tilingData_.set_wOutputTail(splitData_.wOutputTail);
    tilingData_.set_wOutputOuter(splitData_.wOutputOuter);
    tilingData_.set_normalCoreProcessNum(splitData_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(splitData_.tailCoreProcessNum);
    tilingData_.set_usedCoreNum(splitData_.usedCoreNum);
    tilingData_.set_inputBufferSize(splitData_.inputBufferSize);
    tilingData_.set_maxValueBufferSize(splitData_.maxValueBufferSize);
    tilingData_.set_argmaxBufferSize(splitData_.argmaxBufferSize);
    tilingData_.set_isPad(baseData_.isPad);
    tilingData_.set_hDilation(baseData_.hDilation);
    tilingData_.set_wDilation(baseData_.wDilation);
}

ge::graphStatus MaxPoolWithArgmaxV3GatherTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3GatherTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_usedCoreNum());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3GatherTiling, 0);

} // namespace optiling
