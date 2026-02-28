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
 * \file avg_pool_v2_grad_nhwc_tiling.cpp
 * \brief
 */

#include "avg_pool_v2_grad_nhwc_tiling.h"

namespace optiling
{
using namespace AvgPoolV2Grad;

static constexpr uint64_t TILING_KEY_NHWC = 2;
static constexpr uint64_t FORMAT_NHWC = 1;
static constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t DOUBLE_SIZE = 8;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t UB_RESERVED_SIZE = 3072;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;

bool AvgPoolV2GradCommonNHWCTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }

    InitializationVars();

    // all hw overlapped
    if (baseData.hProBatchSize >= inputData.gradShape[H_DIM] && baseData.wProBatchSize >= inputData.gradShape[W_DIM]) {
        return false;
    }

    // ub is not enough
    splitData.nOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = 1;
    splitData.cOutputInner = std::min(inputData.channels, baseData.proDataNumInOneBeat);
    return IsMeetUBSize();
}

void AvgPoolV2GradCommonNHWCTiling::InitializationVars()
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.gradBytes = inputData.dtypeSize;

    baseData.availableUb = ubSize - UB_RESERVED_SIZE;
    baseData.totalCoreNum = coreNum;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNum = baseData.ubBlockSize / baseData.gradBytes;
    baseData.maxDataNumInOneBlock = oneBlockNum;
    baseData.proDataNumInOneBeat = baseData.vRegSize / baseData.ubBlockSize * oneBlockNum;
    baseData.moveDataNumCacheLine = CACHE_LINE_SIZE / baseData.gradBytes;

    baseData.isPad = 0;
    if (inputData.pad[TOP_PAD_INDEX] != 0 || inputData.pad[BOTTOM_PAD_INDEX] != 0 || inputData.pad[LEFT_PAD_INDEX] != 0 || inputData.pad[RIGHT_PAD_INDEX] != 0) {
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

uint64_t AvgPoolV2GradCommonNHWCTiling::GetTilingKey() const
{
    uint64_t schMode = TILING_KEY_NHWC;
    uint64_t format = FORMAT_NHWC;
    uint64_t countIncludePad = inputData.countIncludePad;
    uint64_t isPad = 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        schMode, format, static_cast<uint64_t>(inputData.isInt32Meet), isPad,
        static_cast<uint64_t>(splitData.isCheckRange), countIncludePad, static_cast<uint64_t>(inputData.hasDivisor)
    );

    return tilingKey;
}

void AvgPoolV2GradCommonNHWCTiling::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData.kernelSize[H_DIM] - 1, inputData.stride[H_DIM]);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData.kernelSize[W_DIM] - 1, inputData.stride[W_DIM]);

    int64_t inputPlaneSizeHW = hInputInner * wInputInner;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * splitData.wOutputInner;
    int64_t cOutputAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.maxDataNumInOneBlock);
    int64_t ncPlaneAlignedSize = cOutputAligned * splitData.nOutputInner;

    splitData.inputGradBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.gradBytes + EXTRA_BUFFER_SIZE;

    splitData.outputBufferSize = ncPlaneAlignedSize * outputPlaneSizeHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.inputGradBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool AvgPoolV2GradCommonNHWCTiling::IsMeetTargetCoreNum() const
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[W_DIM], splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[H_DIM], splitData.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(inputData.batches, splitData.nOutputInner);
    int64_t tmpCOutputOuter = Ops::Base::CeilDiv(inputData.channels, splitData.cOutputInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter * tmpCOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool AvgPoolV2GradCommonNHWCTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AvgPoolV2GradCommonNHWCTiling::TrySplitN()
{
    splitData.wOutputInner = inputData.inputShape[W_DIM];
    splitData.hOutputInner = inputData.inputShape[H_DIM];
    splitData.cOutputInner = inputData.channels;

    splitData.nOutputInner = Ops::Base::CeilDiv(inputData.batches, baseData.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.nOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = inputData.batches;
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
    } else {
        return false;
    }
}

bool AvgPoolV2GradCommonNHWCTiling::TrySplitAlignH()
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = inputData.inputShape[W_DIM];
    splitData.cOutputInner = inputData.channels;

    splitData.hOutputInner = inputData.stride[H_DIM];
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.inputShape[H_DIM] / 2, inputData.stride[H_DIM]);
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

bool AvgPoolV2GradCommonNHWCTiling::TrySplitAlignW()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData.stride[H_DIM];
    splitData.cOutputInner = inputData.channels;

    splitData.wOutputInner = inputData.stride[W_DIM];
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.inputShape[W_DIM] / 2, inputData.stride[W_DIM]);
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

bool AvgPoolV2GradCommonNHWCTiling::TrySplitAlignC()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData.stride[H_DIM];
    splitData.wOutputInner = inputData.stride[W_DIM];

    int64_t tmpCAligned =
        inputData.channels < baseData.moveDataNumCacheLine ? inputData.channels : baseData.moveDataNumCacheLine;
    splitData.cOutputInner = tmpCAligned;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.channels / 2, baseData.moveDataNumCacheLine);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.cOutputInner = mid * baseData.moveDataNumCacheLine;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.cOutputInner = bestSplit * baseData.moveDataNumCacheLine;
        return true;
    } else {
        // hw stride 较大场景 或者 nhwc超小场景  ---> 应该对hw做更小的切分
        return false;
    }
}

void AvgPoolV2GradCommonNHWCTiling::SplitUnalignHWC()
{
    splitData.nOutputInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData.stride[H_DIM];
        splitData.wOutputInner = inputData.stride[W_DIM];
        int64_t tmpCAligned =
            inputData.channels < baseData.moveDataNumCacheLine ? inputData.channels : baseData.moveDataNumCacheLine;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.wOutputInner = inputData.inputShape[W_DIM];
        splitData.hOutputInner = inputData.inputShape[H_DIM];
        splitData.cOutputInner = inputData.channels;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[W_DIM], splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[H_DIM], splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner != 1) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    // NHW全切为1  C 超大场景 或者 NHW超小场景
    if (inputData.channels <= baseData.proDataNumInOneBeat) {
        return;
    } else if (IsMeetUBSize()) {
        splitData.cOutputInner = baseData.proDataNumInOneBeat;
        return;
    } else {
            int64_t left = 1;
            int64_t right = Ops::Base::CeilDiv(inputData.channels / 2, baseData.proDataNumInOneBeat);
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
            return;
        }
}

void AvgPoolV2GradCommonNHWCTiling::DynamicAdjustmentWH()
{
    if (splitData.hOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData.inputShape[W_DIM], splitData.wOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData.inputShape[H_DIM], splitData.hOutputOuter);
    }
}

void AvgPoolV2GradCommonNHWCTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
    if (baseData.isPad == 1 || baseData.isOverlap == 1) {
        splitData.isCheckRange = 1;
    } else if (inputData.ceilMode) {
        int64_t tmpH = (inputData.outShape[H_DIM] - inputData.kernelSize[H_DIM] % inputData.stride[H_DIM]);
        int64_t tmpW = (inputData.outShape[W_DIM] - inputData.kernelSize[W_DIM] % inputData.stride[W_DIM]);
        if (tmpH != 0 || tmpW != 0) {
            splitData.isCheckRange = 1;
        }
    }

    if (TrySplitN()) {
        return;
    }

    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
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

    // 带pad 或者 最小整切仍然不满足条件需要更细粒度切分HWC
    splitData.isCheckRange = 1;
    SplitUnalignHWC();
    return;
}

void AvgPoolV2GradCommonNHWCTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[W_DIM], splitData.wOutputInner);
    int64_t tempWOutputTail = inputData.inputShape[W_DIM] % splitData.wOutputInner;
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.inputShape[H_DIM], splitData.hOutputInner);
    int64_t tempHOutputTail = inputData.inputShape[H_DIM] % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData.batches, splitData.nOutputInner);
    int64_t tempNOutputTail = inputData.batches % splitData.nOutputInner;
    splitData.nOutputTail = tempNOutputTail == 0 ? splitData.nOutputInner : tempNOutputTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData.channels, splitData.cOutputInner);
    int64_t tempCOutputTail = inputData.channels % splitData.cOutputInner;
    splitData.cOutputTail = tempCOutputTail == 0 ? splitData.cOutputInner : tempCOutputTail;
}

void AvgPoolV2GradCommonNHWCTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum =
        splitData.nOutputOuter * splitData.cOutputOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void AvgPoolV2GradCommonNHWCTiling::PrintBaseData() const
{
    OP_LOGD("AvgPoolV2GradNHWC", "[AvgPoolV2GradNHWC] PrintBaseData start running");

    std::ostringstream info;
    info << "baseData.vRegSize: " << baseData.vRegSize << std::endl;
    info << "baseData.ubBlockSize: " << baseData.ubBlockSize << std::endl;

    info << "baseData.gradBytes: " << baseData.gradBytes << std::endl;
    info << "baseData.availableUb: " << baseData.availableUb << std::endl;
    info << "baseData.maxDataNumInOneBlock: " << baseData.maxDataNumInOneBlock << std::endl;
    info << "baseData.proDataNumInOneBeat: " << baseData.proDataNumInOneBeat << std::endl;
    info << "baseData.totalCoreNum: " << baseData.totalCoreNum << std::endl;
    info << "baseData.coreUsedForBestPerformance: " << baseData.coreUsedForBestPerformance << std::endl;

    info << "baseData.isPad: " << baseData.isPad << std::endl;
    info << "baseData.isOverlap: " << baseData.isOverlap << std::endl;
    info << "baseData.hProBatchSize: " << baseData.hProBatchSize << std::endl;
    info << "baseData.wProBatchSize: " << baseData.wProBatchSize << std::endl;
    info << "baseData.moveDataNumCacheLine: " << baseData.moveDataNumCacheLine << std::endl;

    OP_LOGI("AvgPoolV2GradNHWC", "%s", info.str().c_str());
}

void AvgPoolV2GradCommonNHWCTiling::PrintSplitData() const
{
    OP_LOGD("AvgPoolV2GradNHWC", "[AvgPoolV2GradNHWC] PrintSplitData start running");

    std::ostringstream info;
    info << "splitData.isCheckRange: " << splitData.isCheckRange << std::endl;

    info << "splitData.nOutputInner: " << splitData.nOutputInner << std::endl;
    info << "splitData.nOutputTail: " << splitData.nOutputTail << std::endl;
    info << "splitData.nOutputOuter: " << splitData.nOutputOuter << std::endl;

    info << "splitData.hOutputInner: " << splitData.hOutputInner << std::endl;
    info << "splitData.hOutputTail: " << splitData.hOutputTail << std::endl;
    info << "splitData.hOutputOuter: " << splitData.hOutputOuter << std::endl;

    info << "splitData.wOutputInner: " << splitData.wOutputInner << std::endl;
    info << "splitData.wOutputTail: " << splitData.wOutputTail << std::endl;
    info << "splitData.wOutputOuter: " << splitData.wOutputOuter << std::endl;

    info << "splitData.cOutputInner: " << splitData.cOutputInner << std::endl;
    info << "splitData.cOutputTail: " << splitData.cOutputTail << std::endl;
    info << "splitData.cOutputOuter: " << splitData.cOutputOuter << std::endl;

    info << "splitData.normalCoreProcessNum: " << splitData.normalCoreProcessNum << std::endl;
    info << "splitData.tailCoreProcessNum: " << splitData.tailCoreProcessNum << std::endl;
    info << "splitData.usedCoreNum: " << splitData.usedCoreNum << std::endl;
    info << "splitData.totalBaseBlockNum: " << splitData.totalBaseBlockNum << std::endl;

    info << "splitData.outputBufferSize: " << splitData.outputBufferSize << std::endl;
    info << "splitData.inputGradBufferSize: " << splitData.inputGradBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

    OP_LOGI("AvgPoolV2GradNHWC", "%s", info.str().c_str());
}

void AvgPoolV2GradCommonNHWCTiling::SetTilingData()
{
    AvgPoolV2GradNHWCTilingData* tilingData = context_->GetTilingData<AvgPoolV2GradNHWCTilingData>();
    tilingData->hGrad = inputData.gradShape[H_DIM];
    tilingData->wGrad = inputData.gradShape[W_DIM];
    tilingData->cOutput = inputData.channels;
    tilingData->hOutput = inputData.inputShape[H_DIM];
    tilingData->wOutput = inputData.inputShape[W_DIM];
    tilingData->hKernel = inputData.kernelSize[H_DIM];
    tilingData->wKernel = inputData.kernelSize[W_DIM];
    tilingData->hStride = inputData.stride[H_DIM];
    tilingData->wStride = inputData.stride[W_DIM];
    tilingData->padTop = inputData.pad[TOP_PAD_INDEX];
    tilingData->padLeft = inputData.pad[LEFT_PAD_INDEX];
    tilingData->padBottom = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->padRight = inputData.pad[RIGHT_PAD_INDEX];
    tilingData->countIncludePad = inputData.countIncludePad;
    tilingData->divisorOverride = inputData.divisorOverride;

    tilingData->nOutputInner = splitData.nOutputInner;
    tilingData->nOutputTail = splitData.nOutputTail;
    tilingData->nOutputOuter = splitData.nOutputOuter;
    tilingData->hOutputInner = splitData.hOutputInner;
    tilingData->hOutputTail = splitData.hOutputTail;
    tilingData->hOutputOuter = splitData.hOutputOuter;
    tilingData->wOutputInner = splitData.wOutputInner;
    tilingData->wOutputTail = splitData.wOutputTail;
    tilingData->wOutputOuter = splitData.wOutputOuter;
    tilingData->cOutputInner = splitData.cOutputInner;
    tilingData->cOutputTail = splitData.cOutputTail;
    tilingData->cOutputOuter = splitData.cOutputOuter;
    tilingData->normalCoreProcessNum = splitData.normalCoreProcessNum;
    tilingData->tailCoreProcessNum = splitData.tailCoreProcessNum;
    tilingData->usedCoreNum = splitData.usedCoreNum;
    tilingData->outputBufferSize = splitData.outputBufferSize;
    tilingData->inputGradBufferSize = splitData.inputGradBufferSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->tilingKey = GetTilingKey();
}

ge::graphStatus AvgPoolV2GradCommonNHWCTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNHWCTiling::PostTiling()
{
    AvgPoolV2GradNHWCTilingData* tilingData = context_->GetTilingData<AvgPoolV2GradNHWCTilingData>();
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNHWCTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradCommonNHWCTiling::GetWorkspaceSize()
{
    auto sysWorkspace = ASCENDC_TOOLS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolV2GradNHWCTiling::GetPlatformInfo() {
    return GetAvgPoolV2GradPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPoolV2GradNHWCTiling::GetShapeAttrsInfo() {
    return GetAvgPoolV2GradShapeAttrsInfo(context_, inputData);
}

REGISTER_OPS_TILING_TEMPLATE(AvgPoolV2Grad, AvgPoolV2GradNHWCTiling, 3);

}  // namespace optiling
