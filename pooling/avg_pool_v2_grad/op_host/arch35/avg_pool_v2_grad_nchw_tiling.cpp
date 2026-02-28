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
 * \file avg_pool_v2_grad_nchw_tiling.cpp
 * \brief
 */

#include "avg_pool_v2_grad_nchw_tiling.h"

namespace optiling {
using namespace AvgPoolV2Grad;

static constexpr int64_t UB_RESVERVED_SIZE = 3072;
static constexpr uint64_t TILING_KEY_NCHW = 1;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t FORMAT_NCHW = 0;
static constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

void AvgPoolV2GradCommonNCHWTiling::InitializationVars()
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData.dtypeSize;
    baseData.availableUb = static_cast<int64_t>(ubSize) - UB_RESVERVED_SIZE;
    baseData.totalCoreNum = static_cast<int64_t>(coreNum);
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;

    baseData.dataNumInOneBlock = oneBlockNumT1;

    baseData.proDataNumInOneBeat = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT1;
    baseData.inputNCSize = inputData.batches * inputData.channels;

    baseData.isPad = 0;
    if (inputData.pad[TOP_PAD_INDEX] != 0 || inputData.pad[LEFT_PAD_INDEX] != 0 ||
        inputData.pad[BOTTOM_PAD_INDEX] != 0 || inputData.pad[RIGHT_PAD_INDEX] != 0) {
        baseData.isPad = 1;
    }

    baseData.hProBatchSize = 1;
    if (inputData.kernelSize[H_DIM] > inputData.stride[H_DIM]) {
        baseData.hProBatchSize = Ops::Base::CeilDiv(inputData.kernelSize[H_DIM], inputData.stride[H_DIM]);
    }

    baseData.wProBatchSize = 1;
    if (inputData.kernelSize[W_DIM] > inputData.stride[W_DIM]) {
        baseData.wProBatchSize = Ops::Base::CeilDiv(inputData.kernelSize[W_DIM], inputData.stride[W_DIM]);
    }

    baseData.isOverlap = 0;
    if (baseData.wProBatchSize != 1 || baseData.hProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}

bool AvgPoolV2GradCommonNCHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    InitializationVars();
    // all the h and w is overlapped.
    if (baseData.hProBatchSize >= inputData.gradShape[H_DIM] && baseData.wProBatchSize >= inputData.gradShape[W_DIM]) {
        return false;
    }
    // ub is not enough
    splitData.highAxisInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = std::min(inputData.outShape[W_DIM], baseData.proDataNumInOneBeat);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

uint64_t AvgPoolV2GradCommonNCHWTiling::GetTilingKey() const
{
    uint64_t schMode = TILING_KEY_NCHW;
    uint64_t format = FORMAT_NCHW;
    uint64_t isPad = 0;  // tilingkey not use this key
    uint64_t countIncludePad = inputData.countIncludePad;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        schMode, format, static_cast<uint64_t>(inputData.isInt32Meet), isPad,
        static_cast<uint64_t>(splitData.isCheckRange), countIncludePad, static_cast<uint64_t>(inputData.hasDivisor));

    return tilingKey;
}

void AvgPoolV2GradCommonNCHWTiling::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner =
        Ops::Base::CeilDiv(splitData.hOutputInner + inputData.kernelSize[H_DIM] - 1, inputData.stride[H_DIM]);
    int64_t wInputInner =
        Ops::Base::CeilDiv(splitData.wOutputInner + inputData.kernelSize[W_DIM] - 1, inputData.stride[W_DIM]);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.dataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.dataNumInOneBlock);

    int64_t inputPlaneSizeHW = hInputInner * wInputInnerAligned;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.inputBytes;
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * sizeof(float); // 累加需要提高精度

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool AvgPoolV2GradCommonNCHWTiling::IsMeetTargetCoreNum() const
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool AvgPoolV2GradCommonNCHWTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AvgPoolV2GradCommonNCHWTiling::TrySplitNC()
{
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.hOutputInner = inputData.outShape[H_DIM];

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

bool AvgPoolV2GradCommonNCHWTiling::TrySplitAlignH()
{
    splitData.highAxisInner = 1;
    splitData.wOutputInner = inputData.outShape[W_DIM];

    splitData.hOutputInner = inputData.stride[H_DIM];
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.outShape[H_DIM] / 2, inputData.stride[H_DIM]);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.hOutputInner = mid * inputData.stride[H_DIM];

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.hOutputInner = bestSplit * inputData.stride[H_DIM];
        return true;
    } else {
        return false;
    }
}

bool AvgPoolV2GradCommonNCHWTiling::TrySplitAlignW()
{
    splitData.highAxisInner = 1;
    splitData.hOutputInner = inputData.stride[H_DIM];

    splitData.wOutputInner = inputData.stride[W_DIM];
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.outShape[W_DIM] / 2, inputData.stride[W_DIM]);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.wOutputInner = mid * inputData.stride[W_DIM];

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.wOutputInner = bestSplit * inputData.stride[W_DIM];
        return true;
    } else {
        return false;
    }
}

void AvgPoolV2GradCommonNCHWTiling::SplitUnalignHW()
{
    splitData.highAxisInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData.stride[H_DIM];
        splitData.wOutputInner = inputData.stride[W_DIM];
    } else {
        splitData.hOutputInner = inputData.outShape[H_DIM];
        splitData.wOutputInner = inputData.outShape[W_DIM];
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner > baseData.proDataNumInOneBeat) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(inputData.outShape[W_DIM], baseData.proDataNumInOneBeat);
    return;
}

void AvgPoolV2GradCommonNCHWTiling::DynamicAdjustmentWH()
{
    if (splitData.hOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputOuter);
    }
}

void AvgPoolV2GradCommonNCHWTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
    if (baseData.isPad == 1 || baseData.isOverlap == 1) {
        splitData.isCheckRange = 1;
    } else if (inputData.ceilMode) {
        // pad = 0 but kernelsize != kH * kW
        int64_t tmpH = (inputData.outShape[H_DIM] - inputData.kernelSize[H_DIM]) % inputData.stride[H_DIM];
        int64_t tmpW = (inputData.outShape[W_DIM] - inputData.kernelSize[W_DIM]) % inputData.stride[W_DIM];
        if (tmpH != 0 || tmpW != 0) {
            splitData.isCheckRange = 1;
        }
    }

    if (TrySplitNC()) {
        return;
    }

    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        if (TrySplitAlignH()) {
            return;
        }

        if (TrySplitAlignW()) {
            return;
        }
    }

    // 带pad 或者overlap 或者 最小整切仍然不满足条件需要更细粒度切分HW
    splitData.isCheckRange = 1;
    SplitUnalignHW();
    return;
}

void AvgPoolV2GradCommonNCHWTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t tempWOutputTail = inputData.outShape[W_DIM] % splitData.wOutputInner;
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t tempHOutputTail = inputData.outShape[H_DIM] % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    int64_t tempHighAxisTail = baseData.inputNCSize % splitData.highAxisInner;
    splitData.highAxisTail = tempHighAxisTail == 0 ? splitData.highAxisInner : tempHighAxisTail;
}

void AvgPoolV2GradCommonNCHWTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void AvgPoolV2GradCommonNCHWTiling::PrintBaseData() const
{
    OP_LOGI(context_->GetNodeName(), "PrintBaseData start running");
    std::ostringstream info;
    info << "baseData.vRegSize: " << baseData.vRegSize;
    info << ", baseData.ubBlockSize: " << baseData.ubBlockSize;
    info << ", baseData.inputBytes: " << baseData.inputBytes;
    info << ", baseData.availableUb: " << baseData.availableUb;
    info << ", baseData.dataNumInOneBlock: " << baseData.dataNumInOneBlock;
    info << ", baseData.proDataNumInOneBeat: " << baseData.proDataNumInOneBeat;
    info << ", baseData.totalCoreNum: " << baseData.totalCoreNum;
    info << ", baseData.coreUsedForBestPerformance: " << baseData.coreUsedForBestPerformance;
    info << ", baseData.isPad: " << baseData.isPad;
    info << ", baseData.isOverlap: " << baseData.isOverlap;
    info << ", baseData.hProBatchSize: " << baseData.hProBatchSize;
    info << ", baseData.wProBatchSize: " << baseData.wProBatchSize;
    info << ", baseData.inputNCSize: " << baseData.inputNCSize;
    info << ", padTopH: " << inputData.pad[TOP_PAD_INDEX];
    info << ", padDownH: " << inputData.pad[BOTTOM_PAD_INDEX];
    info << ", padLfetW: " << inputData.pad[LEFT_PAD_INDEX];
    info << ", padRightW: " << inputData.pad[RIGHT_PAD_INDEX];
    info << ", divisorOverride: " << inputData.divisorOverride;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void AvgPoolV2GradCommonNCHWTiling::PrintSplitData() const
{
    OP_LOGI(context_->GetNodeName(), "PrintInputData start running");
    std::ostringstream info;
    info << "splitData.isCheckRange: " << splitData.isCheckRange;

    info << ", splitData.highAxisInner: " << splitData.highAxisInner;
    info << ", splitData.highAxisTail: " << splitData.highAxisTail;
    info << ", splitData.highAxisOuter: " << splitData.highAxisOuter;

    info << ", splitData.hOutputInner: " << splitData.hOutputInner;
    info << ", splitData.hOutputTail: " << splitData.hOutputTail;
    info << ", splitData.hOutputOuter: " << splitData.hOutputOuter;

    info << ", splitData.wOutputInner: " << splitData.wOutputInner;
    info << ", splitData.wOutputTail: " << splitData.wOutputTail;
    info << ", splitData.wOutputOuter: " << splitData.wOutputOuter;

    info << ", splitData.normalCoreProcessNum: " << splitData.normalCoreProcessNum;
    info << ", splitData.tailCoreProcessNum: " << splitData.tailCoreProcessNum;
    info << ", splitData.usedCoreNum: " << splitData.usedCoreNum;
    info << ", splitData.totalBaseBlockNum: " << splitData.totalBaseBlockNum;

    info << ", splitData.outputBufferSize: " << splitData.outputBufferSize;
    info << ", splitData.gradBufferSize: " << splitData.gradBufferSize;
    info << ", splitData.totalBufferSize: " << splitData.totalBufferSize;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void AvgPoolV2GradCommonNCHWTiling::SetTilingData()
{
    AvgPoolV2GradNCHWTilingData* tilingData = context_->GetTilingData<AvgPoolV2GradNCHWTilingData>();
    tilingData->hGrad = inputData.gradShape[H_DIM];
    tilingData->wGrad = inputData.gradShape[W_DIM];
    tilingData->hOutput = inputData.outShape[H_DIM];
    tilingData->wOutput = inputData.outShape[W_DIM];
    tilingData->hKernel = inputData.kernelSize[H_DIM];
    tilingData->wKernel = inputData.kernelSize[W_DIM];
    tilingData->hStride = inputData.stride[H_DIM];
    tilingData->wStride = inputData.stride[W_DIM];
    tilingData->padTopH = inputData.pad[TOP_PAD_INDEX];
    tilingData->padDownH = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->padLeftW = inputData.pad[LEFT_PAD_INDEX];
    tilingData->padRightW = inputData.pad[RIGHT_PAD_INDEX];
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
    tilingData->outputBufferSize = splitData.outputBufferSize;
    tilingData->gradBufferSize = splitData.gradBufferSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->divisorOverride = inputData.divisorOverride;
}

ge::graphStatus AvgPoolV2GradCommonNCHWTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNCHWTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNCHWTiling::GetWorkspaceSize()
{
    auto sysWorkspace = ASCENDC_TOOLS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNCHWTiling::PostTiling()
{
    context_->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

//////////////////////////////// AvgPoolV2GradNCHWTiling /////////////////////////////////
ge::graphStatus AvgPoolV2GradNCHWTiling::GetPlatformInfo()
{
    return GetAvgPoolV2GradPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolV2GradNCHWTiling::GetShapeAttrsInfo()
{
    ge::graphStatus ret = GetAvgPoolV2GradShapeAttrsInfo(context_, inputData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ret;
}

REGISTER_TILING_TEMPLATE("AvgPoolV2Grad", AvgPoolV2GradNCHWTiling, 0);

} // namespace optiling
