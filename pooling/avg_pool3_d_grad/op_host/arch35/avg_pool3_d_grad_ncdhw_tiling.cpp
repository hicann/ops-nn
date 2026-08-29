/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ncdhw_tiling.cpp
 * \brief NCDHW scheme tiling for 3D average pooling backward (arch35).
 *        Logic modeled on avg_pool_v2_grad_nchw_tiling.cpp, extended to D/H/W.
 */

#include <sstream>
#include "op_host/tiling_templates_registry.h"
#include "avg_pool3_d_grad_ncdhw_tiling.h"

namespace optiling {
using namespace AvgPool3DGrad;

static constexpr int64_t UB_HELPER_SIZE = 3072;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t TILING_KEY_NCDHW = 1;
static constexpr uint64_t FORMAT_NCDHW = 0;
static constexpr int64_t VL_FACTOR = 4;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t BANK_FACTOR = 128;

void AvgPool3DGradNCDHWTiling::InitializationVars()
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData.dtypeSize;
    baseData.availableUb = static_cast<int64_t>(ubSize) - UB_HELPER_SIZE;
    baseData.totalCoreNum = static_cast<int64_t>(coreNum);
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNum = baseData.ubBlockSize / baseData.inputBytes;
    baseData.dataNumInOneBlock = oneBlockNum;
    baseData.proDataNumInOneBeat = baseData.vRegSize / baseData.ubBlockSize * oneBlockNum;
    baseData.inputNCSize = inputData.batches * inputData.channels; // NCDHW: batches = N*C, channels = 1

    baseData.isPad = 0;
    if (inputData.pad[FRONT_PAD_INDEX] != 0 || inputData.pad[BACKEND_PAD_INDEX] != 0 ||
        inputData.pad[TOP_PAD_INDEX] != 0 || inputData.pad[BOTTOM_PAD_INDEX] != 0 ||
        inputData.pad[LEFT_PAD_INDEX] != 0 || inputData.pad[RIGHT_PAD_INDEX] != 0) {
        baseData.isPad = 1;
    }

    baseData.dProBatchSize = 1;
    if (inputData.kernelSize[D_DIM] > inputData.stride[D_DIM]) {
        baseData.dProBatchSize = Ops::Base::CeilDiv(inputData.kernelSize[D_DIM], inputData.stride[D_DIM]);
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
    if (baseData.dProBatchSize != 1 || baseData.hProBatchSize != 1 || baseData.wProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}

bool AvgPool3DGradNCDHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }
    InitializationVars();
    // all the d/h/w are overlapped: fall back to simt.
    if (baseData.dProBatchSize >= inputData.gradShape[D_DIM] && baseData.hProBatchSize >= inputData.gradShape[H_DIM] &&
        baseData.wProBatchSize >= inputData.gradShape[W_DIM]) {
        return false;
    }
    splitData.highAxisInner = 1;
    splitData.dOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = std::min(inputData.outShape[W_DIM], baseData.proDataNumInOneBeat);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

uint64_t AvgPool3DGradNCDHWTiling::GetTilingKey() const
{
    uint64_t schMode = TILING_KEY_NCDHW;
    uint64_t format = FORMAT_NCDHW;
    uint64_t isPad = 0; // not used by this key
    uint64_t countIncludePad = inputData.countIncludePad;
    return GET_TPL_TILING_KEY(schMode, format, static_cast<uint64_t>(inputData.isInt32Meet), isPad,
                              static_cast<uint64_t>(splitData.isCheckRange), countIncludePad,
                              static_cast<uint64_t>(inputData.hasDivisor));
}

void AvgPool3DGradNCDHWTiling::DoBufferCalculate()
{
    splitData.dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner + inputData.kernelSize[D_DIM] - 1,
                                               inputData.stride[D_DIM]);
    splitData.hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData.kernelSize[H_DIM] - 1,
                                               inputData.stride[H_DIM]);
    splitData.wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData.kernelSize[W_DIM] - 1,
                                               inputData.stride[W_DIM]);

    int64_t wInputAligned = Ops::Base::CeilAlign(splitData.wInputInner, baseData.dataNumInOneBlock);
    int64_t wOutputAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.dataNumInOneBlock);
    int64_t inputPlaneSize = splitData.dInputInner * splitData.hInputInner * wInputAligned;
    int64_t outputPlaneSize = splitData.dOutputInner * splitData.hOutputInner * wOutputAligned;

    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSize * baseData.inputBytes;
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSize * sizeof(float); // accumulate in fp32
    splitData.totalBufferSize = (splitData.gradBufferSize + splitData.outputBufferSize) * DOUBLE_BUFFER;
}

bool AvgPool3DGradNCDHWTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    if (baseData.inputBytes == ge::GetSizeByDataType(ge::DT_FLOAT16)) {
        return splitData.totalBufferSize <= baseData.availableUb &&
               splitData.gradBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes;
    }
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AvgPool3DGradNCDHWTiling::IsMeetTargetCoreNum() const
{
    int64_t wOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t hOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t dOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);
    int64_t highOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    return wOuter * hOuter * dOuter * highOuter >= baseData.coreUsedForBestPerformance;
}

bool AvgPool3DGradNCDHWTiling::TrySplitNC()
{
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.hOutputInner = inputData.outShape[H_DIM];
    splitData.dOutputInner = inputData.outShape[D_DIM];

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

bool AvgPool3DGradNCDHWTiling::TrySplitAlignD()
{
    splitData.highAxisInner = 1;
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.hOutputInner = inputData.outShape[H_DIM];

    splitData.dOutputInner = inputData.stride[D_DIM];
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.outShape[D_DIM] / 2, inputData.stride[D_DIM]);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.dOutputInner = mid * inputData.stride[D_DIM];
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.dOutputInner = bestSplit * inputData.stride[D_DIM];
        return true;
    } else {
        return false;
    }
}

bool AvgPool3DGradNCDHWTiling::TrySplitAlignH()
{
    splitData.highAxisInner = 1;
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.dOutputInner = inputData.stride[D_DIM];

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

bool AvgPool3DGradNCDHWTiling::TrySplitAlignW()
{
    splitData.highAxisInner = 1;
    splitData.dOutputInner = inputData.stride[D_DIM];
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

void AvgPool3DGradNCDHWTiling::SplitUnalignDHW()
{
    splitData.highAxisInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.dOutputInner = inputData.stride[D_DIM];
        splitData.hOutputInner = inputData.stride[H_DIM];
        splitData.wOutputInner = inputData.stride[W_DIM];
    } else {
        splitData.dOutputInner = inputData.outShape[D_DIM];
        splitData.hOutputInner = inputData.outShape[H_DIM];
        splitData.wOutputInner = inputData.outShape[W_DIM];
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);

    while (splitData.dOutputInner != 1 || splitData.hOutputInner != 1 ||
           splitData.wOutputInner > baseData.proDataNumInOneBeat) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDHW();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(inputData.outShape[W_DIM], baseData.proDataNumInOneBeat);
    return;
}

void AvgPool3DGradNCDHWTiling::DynamicAdjustmentDHW()
{
    if (splitData.hOutputInner == 1 && splitData.dOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputOuter);
    } else if (splitData.hOutputInner == 1) {
        splitData.dOutputOuter++;
        splitData.dOutputInner = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputOuter);
    }
}

void AvgPool3DGradNCDHWTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
    splitData.isStrideAligned = 0;
    if (baseData.isPad == 1 || baseData.isOverlap == 1) {
        splitData.isCheckRange = 1;
    } else if (inputData.ceilMode) {
        int64_t tmpD = (inputData.outShape[D_DIM] - inputData.kernelSize[D_DIM]) % inputData.stride[D_DIM];
        int64_t tmpH = (inputData.outShape[H_DIM] - inputData.kernelSize[H_DIM]) % inputData.stride[H_DIM];
        int64_t tmpW = (inputData.outShape[W_DIM] - inputData.kernelSize[W_DIM]) % inputData.stride[W_DIM];
        if (tmpD != 0 || tmpH != 0 || tmpW != 0) {
            splitData.isCheckRange = 1;
        }
    }

    if (TrySplitNC()) {
        return;
    }

    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        if (TrySplitAlignD()) {
            splitData.isStrideAligned = 1;
            return;
        }
        if (TrySplitAlignH()) {
            splitData.isStrideAligned = 1;
            return;
        }
        if (TrySplitAlignW()) {
            splitData.isStrideAligned = 1;
            return;
        }
    }

    splitData.isCheckRange = 1;
    SplitUnalignDHW();
    return;
}

void AvgPool3DGradNCDHWTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
}

void AvgPool3DGradNCDHWTiling::SearchOuterSingle(int64_t& inner, int64_t step)
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

void AvgPool3DGradNCDHWTiling::AdjustInnerSplitForMultiCore()
{
    DoBlockTiling();
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        return;
    }

    SearchOuterSingle(splitData.highAxisInner, 1);
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        DoBufferCalculate();
        return;
    }

    SearchOuterSingle(splitData.dOutputInner, splitData.isStrideAligned ? inputData.stride[D_DIM] : 1);
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        DoBufferCalculate();
        return;
    }

    SearchOuterSingle(splitData.hOutputInner, splitData.isStrideAligned ? inputData.stride[H_DIM] : 1);
    if (splitData.usedCoreNum >= baseData.totalCoreNum) {
        DoBufferCalculate();
        return;
    }

    SearchOuterSingle(splitData.wOutputInner, splitData.isStrideAligned ? inputData.stride[W_DIM] : 1);
    DoBufferCalculate();
}

void AvgPool3DGradNCDHWTiling::DoBlockTiling()
{
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t wTail = inputData.outShape[W_DIM] % splitData.wOutputInner;
    splitData.wOutputTail = wTail == 0 ? splitData.wOutputInner : wTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t hTail = inputData.outShape[H_DIM] % splitData.hOutputInner;
    splitData.hOutputTail = hTail == 0 ? splitData.hOutputInner : hTail;

    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);
    int64_t dTail = inputData.outShape[D_DIM] % splitData.dOutputInner;
    splitData.dOutputTail = dTail == 0 ? splitData.dOutputInner : dTail;

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    int64_t highTail = baseData.inputNCSize % splitData.highAxisInner;
    splitData.highAxisTail = highTail == 0 ? splitData.highAxisInner : highTail;

    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.dOutputOuter * splitData.hOutputOuter *
                                  splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

ge::graphStatus AvgPool3DGradNCDHWTiling::SetTilingData()
{
    AvgPool3DGradNCDHWTilingData* tilingData = context_->GetTilingData<AvgPool3DGradNCDHWTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->ncTotal = baseData.inputNCSize;
    tilingData->dGrad = inputData.gradShape[D_DIM];
    tilingData->hGrad = inputData.gradShape[H_DIM];
    tilingData->wGrad = inputData.gradShape[W_DIM];
    tilingData->dOutput = inputData.outShape[D_DIM];
    tilingData->hOutput = inputData.outShape[H_DIM];
    tilingData->wOutput = inputData.outShape[W_DIM];
    tilingData->dKernel = inputData.kernelSize[D_DIM];
    tilingData->hKernel = inputData.kernelSize[H_DIM];
    tilingData->wKernel = inputData.kernelSize[W_DIM];
    tilingData->dStride = inputData.stride[D_DIM];
    tilingData->hStride = inputData.stride[H_DIM];
    tilingData->wStride = inputData.stride[W_DIM];
    tilingData->padFrontD = inputData.pad[FRONT_PAD_INDEX];
    tilingData->padBackD = inputData.pad[BACKEND_PAD_INDEX];
    tilingData->padTopH = inputData.pad[TOP_PAD_INDEX];
    tilingData->padDownH = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->padLeftW = inputData.pad[LEFT_PAD_INDEX];
    tilingData->padRightW = inputData.pad[RIGHT_PAD_INDEX];
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
    tilingData->gradBufferSize = splitData.gradBufferSize;
    tilingData->dProBatchSize = baseData.dProBatchSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->divisorOverride = inputData.divisorOverride;
    tilingData->countIncludePad = inputData.countIncludePad;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradNCDHWTiling::DoOpTiling()
{
    DoUBTiling();
    AdjustInnerSplitForMultiCore();

    int64_t wBatchCnt = std::min(splitData.wInputInner, inputData.gradShape[W_DIM]) / baseData.wProBatchSize;
    wBatchCnt = wBatchCnt > 1 ? wBatchCnt : 1;
    int64_t vlLen = baseData.vRegSize / sizeof(float);

    if (wBatchCnt <= vlLen / DOUBLE) {
        OP_CHECK_IF(baseData.isOverlap,
                    OP_LOGI(context_->GetNodeName(), "ncdhw template is not capable for overlap case."),
                    return ge::GRAPH_PARAM_INVALID);

        int64_t hBatchCnt = std::min(splitData.hInputInner, inputData.gradShape[H_DIM]) / baseData.hProBatchSize;
        hBatchCnt = hBatchCnt > 1 ? hBatchCnt : 1;

        if (wBatchCnt * hBatchCnt <= vlLen / DOUBLE) {
            int64_t dBatchCnt = std::min(splitData.dInputInner, inputData.gradShape[D_DIM]) / baseData.dProBatchSize;
            dBatchCnt = dBatchCnt > 1 ? dBatchCnt : 1;

            int64_t dhwBatchCnt = dBatchCnt * hBatchCnt * wBatchCnt;
            OP_CHECK_IF(
                dhwBatchCnt > vlLen,
                OP_LOGI(context_->GetNodeName(),
                        "ncdhw template not capable, dhwBatchCnt %ld exceeds VReg capacity %ld.", dhwBatchCnt, vlLen),
                return ge::GRAPH_PARAM_INVALID);

            int64_t allGatherCnt = dhwBatchCnt * splitData.highAxisInner;
            OP_CHECK_IF(
                allGatherCnt <= vlLen / VL_FACTOR,
                OP_LOGI(context_->GetNodeName(),
                        "ncdhw template not capable, allGatherCnt: %ld, dBatch:%ld, hBatch:%ld, wBatch:%ld, high:%ld.",
                        allGatherCnt, dBatchCnt, hBatchCnt, wBatchCnt, splitData.highAxisInner),
                return ge::GRAPH_PARAM_INVALID);
        }
    }

    bool bankConflictGrad = (baseData.wProBatchSize * baseData.inputBytes) % BANK_FACTOR == 0;
    bool bankConflictOut = (baseData.wProBatchSize * inputData.stride[W_DIM] * sizeof(float)) % BANK_FACTOR == 0;
    OP_CHECK_IF(bankConflictGrad || bankConflictOut,
                OP_LOGI(context_->GetNodeName(), "ncdhw template not capable because of bank conflict."),
                return ge::GRAPH_PARAM_INVALID);

    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

void AvgPool3DGradNCDHWTiling::PrintBaseData() const
{
    OP_LOGI(context_->GetNodeName(), "PrintBaseData start running");
    std::ostringstream info;
    info << "vRegSize: " << baseData.vRegSize;
    info << ", ubBlockSize: " << baseData.ubBlockSize;
    info << ", inputBytes: " << baseData.inputBytes;
    info << ", availableUb: " << baseData.availableUb;
    info << ", dataNumInOneBlock: " << baseData.dataNumInOneBlock;
    info << ", proDataNumInOneBeat: " << baseData.proDataNumInOneBeat;
    info << ", totalCoreNum: " << baseData.totalCoreNum;
    info << ", coreUsedForBestPerformance: " << baseData.coreUsedForBestPerformance;
    info << ", isPad: " << baseData.isPad;
    info << ", isOverlap: " << baseData.isOverlap;
    info << ", dProBatchSize: " << baseData.dProBatchSize;
    info << ", hProBatchSize: " << baseData.hProBatchSize;
    info << ", wProBatchSize: " << baseData.wProBatchSize;
    info << ", inputNCSize: " << baseData.inputNCSize;
    info << ", padFrontD: " << inputData.pad[FRONT_PAD_INDEX];
    info << ", padBackD: " << inputData.pad[BACKEND_PAD_INDEX];
    info << ", padTopH: " << inputData.pad[TOP_PAD_INDEX];
    info << ", padDownH: " << inputData.pad[BOTTOM_PAD_INDEX];
    info << ", padLeftW: " << inputData.pad[LEFT_PAD_INDEX];
    info << ", padRightW: " << inputData.pad[RIGHT_PAD_INDEX];
    info << ", divisorOverride: " << inputData.divisorOverride;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void AvgPool3DGradNCDHWTiling::PrintSplitData() const
{
    OP_LOGI(context_->GetNodeName(), "PrintSplitData start running");
    std::ostringstream info;
    info << "isCheckRange: " << splitData.isCheckRange;
    info << ", highAxisInner: " << splitData.highAxisInner;
    info << ", highAxisTail: " << splitData.highAxisTail;
    info << ", highAxisOuter: " << splitData.highAxisOuter;
    info << ", dOutputInner: " << splitData.dOutputInner;
    info << ", dOutputTail: " << splitData.dOutputTail;
    info << ", dOutputOuter: " << splitData.dOutputOuter;
    info << ", hOutputInner: " << splitData.hOutputInner;
    info << ", hOutputTail: " << splitData.hOutputTail;
    info << ", hOutputOuter: " << splitData.hOutputOuter;
    info << ", wOutputInner: " << splitData.wOutputInner;
    info << ", wOutputTail: " << splitData.wOutputTail;
    info << ", wOutputOuter: " << splitData.wOutputOuter;
    info << ", normalCoreProcessNum: " << splitData.normalCoreProcessNum;
    info << ", tailCoreProcessNum: " << splitData.tailCoreProcessNum;
    info << ", usedCoreNum: " << splitData.usedCoreNum;
    info << ", totalBaseBlockNum: " << splitData.totalBaseBlockNum;
    info << ", outputBufferSize: " << splitData.outputBufferSize;
    info << ", gradBufferSize: " << splitData.gradBufferSize;
    info << ", totalBufferSize: " << splitData.totalBufferSize;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus AvgPool3DGradNCDHWTiling::PostTiling()
{
    context_->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradNCDHWTiling::GetShapeAttrsInfo()
{
    auto ret = AvgPool3DGradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        OP_LOGI(context_->GetNodeName(), "ncdhw scheme not capable, expected NCDHW format");
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AvgPool3DGrad, AvgPool3DGradNCDHWTiling, 1);

} // namespace optiling
