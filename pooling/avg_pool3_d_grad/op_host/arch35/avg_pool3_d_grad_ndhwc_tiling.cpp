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
 * \file avg_pool3_d_grad_ndhwc_tiling.cpp
 * \brief NDHWC scheme tiling for 3D average pooling backward (arch35).
 *        Logic modeled on avg_pool_v2_grad_nhwc_tiling.cpp, extended to D/H/W.
 */

#include "op_host/tiling_templates_registry.h"
#include "avg_pool3_d_grad_ndhwc_tiling.h"

namespace optiling {
using namespace AvgPool3DGrad;

static constexpr uint64_t TILING_KEY_NDHWC = 2;
static constexpr uint64_t FORMAT_NDHWC = 1;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t UB_HELP_SIZE = 10240;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t BATCH_LIMIT = 256;

bool AvgPool3DGradNDHWCTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NDHWC) {
        return false;
    }
    InitializationVars();
    // all dhw overlapped: fall back to simt.
    if (baseData.dProBatchSize >= inputData.gradShape[D_DIM] && baseData.hProBatchSize >= inputData.gradShape[H_DIM] &&
        baseData.wProBatchSize >= inputData.gradShape[W_DIM]) {
        return false;
    }

    int64_t batchSize = baseData.dProBatchSize * baseData.hProBatchSize * baseData.wProBatchSize;
    if (batchSize >= BATCH_LIMIT) {
        OP_LOGI(context_->GetNodeName(), "The batch is too large.");
        return false;
    }
    splitData.nOutputInner = 1;
    splitData.dOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = 1;
    splitData.cOutputInner = std::min(inputData.channels, baseData.proDataNumInOneBeat);
    return IsMeetUBSize();
}

void AvgPool3DGradNDHWCTiling::InitializationVars()
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData.dtypeSize;
    baseData.availableUb = static_cast<int64_t>(ubSize) - UB_HELP_SIZE;
    baseData.totalCoreNum = static_cast<int64_t>(coreNum);
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNum = baseData.ubBlockSize / baseData.inputBytes;
    baseData.dataNumInOneBlock = oneBlockNum;
    baseData.proDataNumInOneBeat = baseData.vRegSize / baseData.ubBlockSize * oneBlockNum;
    baseData.moveDataNumCacheLine = CACHE_LINE_SIZE / baseData.inputBytes;

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

uint64_t AvgPool3DGradNDHWCTiling::GetTilingKey() const
{
    uint64_t schMode = TILING_KEY_NDHWC;
    uint64_t format = FORMAT_NDHWC;
    uint64_t countIncludePad = inputData.countIncludePad;
    uint64_t isPad = 0;
    return GET_TPL_TILING_KEY(schMode, format, static_cast<uint64_t>(inputData.isInt32Meet), isPad,
                              static_cast<uint64_t>(splitData.isCheckRange), countIncludePad,
                              static_cast<uint64_t>(inputData.hasDivisor));
}

void AvgPool3DGradNDHWCTiling::DoBufferCalculate()
{
    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner + inputData.kernelSize[D_DIM] - 1,
                                             inputData.stride[D_DIM]);
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData.kernelSize[H_DIM] - 1,
                                             inputData.stride[H_DIM]);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData.kernelSize[W_DIM] - 1,
                                             inputData.stride[W_DIM]);

    int64_t inputPlaneSizeDHW = dInputInner * hInputInner * wInputInner;
    int64_t outputPlaneSizeDHW = splitData.dOutputInner * splitData.hOutputInner * splitData.wOutputInner;
    int64_t cOutputAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.dataNumInOneBlock);
    int64_t ncPlaneAlignedSize = cOutputAligned * splitData.nOutputInner;

    splitData.inputGradBufferSize = ncPlaneAlignedSize * inputPlaneSizeDHW * baseData.inputBytes + EXTRA_BUFFER_SIZE;
    splitData.outputBufferSize = ncPlaneAlignedSize * outputPlaneSizeDHW * FLOAT32_SIZE;
    splitData.totalBufferSize = (splitData.inputGradBufferSize + splitData.outputBufferSize) * DOUBLE_BUFFER;
}

bool AvgPool3DGradNDHWCTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    if (baseData.inputBytes == ge::GetSizeByDataType(ge::DT_FLOAT16)) {
        return splitData.totalBufferSize <= baseData.availableUb &&
               splitData.inputGradBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes;
    }
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool AvgPool3DGradNDHWCTiling::IsMeetTargetCoreNum() const
{
    int64_t wOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t hOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t dOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);
    int64_t nOuter = Ops::Base::CeilDiv(inputData.batches, splitData.nOutputInner);
    int64_t cOuter = Ops::Base::CeilDiv(inputData.channels, splitData.cOutputInner);
    return wOuter * hOuter * dOuter * nOuter * cOuter >= baseData.coreUsedForBestPerformance;
}

bool AvgPool3DGradNDHWCTiling::TrySplitN()
{
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.hOutputInner = inputData.outShape[H_DIM];
    splitData.dOutputInner = inputData.outShape[D_DIM];
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

bool AvgPool3DGradNDHWCTiling::TrySplitAlignD()
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.hOutputInner = inputData.outShape[H_DIM];
    splitData.cOutputInner = inputData.channels;

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

bool AvgPool3DGradNDHWCTiling::TrySplitAlignH()
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = inputData.outShape[W_DIM];
    splitData.dOutputInner = inputData.stride[D_DIM];
    splitData.cOutputInner = inputData.channels;

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

bool AvgPool3DGradNDHWCTiling::TrySplitAlignW()
{
    splitData.nOutputInner = 1;
    splitData.dOutputInner = inputData.stride[D_DIM];
    splitData.hOutputInner = inputData.stride[H_DIM];
    splitData.cOutputInner = inputData.channels;

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

bool AvgPool3DGradNDHWCTiling::TrySplitAlignC()
{
    splitData.nOutputInner = 1;
    splitData.dOutputInner = inputData.stride[D_DIM];
    splitData.hOutputInner = inputData.stride[H_DIM];
    splitData.wOutputInner = inputData.stride[W_DIM];

    int64_t tmpCAligned = inputData.channels < baseData.moveDataNumCacheLine ? inputData.channels :
                                                                               baseData.moveDataNumCacheLine;
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
        return false;
    }
}

void AvgPool3DGradNDHWCTiling::SplitUnalignDHWC()
{
    splitData.nOutputInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.dOutputInner = inputData.stride[D_DIM];
        splitData.hOutputInner = inputData.stride[H_DIM];
        splitData.wOutputInner = inputData.stride[W_DIM];
        int64_t tmpCAligned = inputData.channels < baseData.moveDataNumCacheLine ? inputData.channels :
                                                                                   baseData.moveDataNumCacheLine;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.dOutputInner = inputData.outShape[D_DIM];
        splitData.hOutputInner = inputData.outShape[H_DIM];
        splitData.wOutputInner = inputData.outShape[W_DIM];
        splitData.cOutputInner = inputData.channels;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);

    while (splitData.dOutputInner != 1 || splitData.hOutputInner != 1 || splitData.wOutputInner != 1) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentDHW();
        } else {
            return;
        }
    }

    // DHW all split to 1. C 超大场景 或 DHW 超小场景.
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

void AvgPool3DGradNDHWCTiling::DynamicAdjustmentDHW()
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

void AvgPool3DGradNDHWCTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
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

    if (TrySplitN()) {
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
        if (TrySplitAlignC()) {
            return;
        }
    }

    splitData.isCheckRange = 1;
    SplitUnalignDHWC();
    return;
}

void AvgPool3DGradNDHWCTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.outShape[W_DIM], splitData.wOutputInner);
    int64_t wTail = inputData.outShape[W_DIM] % splitData.wOutputInner;
    splitData.wOutputTail = wTail == 0 ? splitData.wOutputInner : wTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.outShape[H_DIM], splitData.hOutputInner);
    int64_t hTail = inputData.outShape[H_DIM] % splitData.hOutputInner;
    splitData.hOutputTail = hTail == 0 ? splitData.hOutputInner : hTail;

    splitData.dOutputOuter = Ops::Base::CeilDiv(inputData.outShape[D_DIM], splitData.dOutputInner);
    int64_t dTail = inputData.outShape[D_DIM] % splitData.dOutputInner;
    splitData.dOutputTail = dTail == 0 ? splitData.dOutputInner : dTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData.batches, splitData.nOutputInner);
    int64_t nTail = inputData.batches % splitData.nOutputInner;
    splitData.nOutputTail = nTail == 0 ? splitData.nOutputInner : nTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData.channels, splitData.cOutputInner);
    int64_t cTail = inputData.channels % splitData.cOutputInner;
    splitData.cOutputTail = cTail == 0 ? splitData.cOutputInner : cTail;
}

void AvgPool3DGradNDHWCTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.nOutputOuter * splitData.cOutputOuter * splitData.dOutputOuter *
                                  splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void AvgPool3DGradNDHWCTiling::SetTilingData()
{
    AvgPool3DGradNDHWCTilingData* tilingData = context_->GetTilingData<AvgPool3DGradNDHWCTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "tilingData is nullptr!");
        return;
    }
    tilingData->nTotal = inputData.batches;
    tilingData->dGrad = inputData.gradShape[D_DIM];
    tilingData->hGrad = inputData.gradShape[H_DIM];
    tilingData->wGrad = inputData.gradShape[W_DIM];
    tilingData->cOutput = inputData.channels;
    tilingData->dOutput = inputData.outShape[D_DIM];
    tilingData->hOutput = inputData.outShape[H_DIM];
    tilingData->wOutput = inputData.outShape[W_DIM];
    tilingData->dKernel = inputData.kernelSize[D_DIM];
    tilingData->hKernel = inputData.kernelSize[H_DIM];
    tilingData->wKernel = inputData.kernelSize[W_DIM];
    tilingData->dStride = inputData.stride[D_DIM];
    tilingData->hStride = inputData.stride[H_DIM];
    tilingData->wStride = inputData.stride[W_DIM];
    tilingData->padFront = inputData.pad[FRONT_PAD_INDEX];
    tilingData->padBack = inputData.pad[BACKEND_PAD_INDEX];
    tilingData->padTop = inputData.pad[TOP_PAD_INDEX];
    tilingData->padBottom = inputData.pad[BOTTOM_PAD_INDEX];
    tilingData->padLeft = inputData.pad[LEFT_PAD_INDEX];
    tilingData->padRight = inputData.pad[RIGHT_PAD_INDEX];
    tilingData->nOutputInner = splitData.nOutputInner;
    tilingData->nOutputTail = splitData.nOutputTail;
    tilingData->nOutputOuter = splitData.nOutputOuter;
    tilingData->dOutputInner = splitData.dOutputInner;
    tilingData->dOutputTail = splitData.dOutputTail;
    tilingData->dOutputOuter = splitData.dOutputOuter;
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
    tilingData->dProBatchSize = baseData.dProBatchSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->divisorOverride = inputData.divisorOverride;
    tilingData->countIncludePad = inputData.countIncludePad;
    tilingData->tilingKey = GetTilingKey();
}

ge::graphStatus AvgPool3DGradNDHWCTiling::DoOpTiling()
{
    InitializationVars();
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradNDHWCTiling::PostTiling()
{
    context_->SetBlockDim(splitData.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradNDHWCTiling::GetShapeAttrsInfo()
{
    auto ret = AvgPool3DGradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (inputData.inputFormat != ge::Format::FORMAT_NDHWC) {
        OP_LOGI(context_->GetNodeName(), "ndhwc scheme not capable, expected NDHWC format");
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AvgPool3DGrad, AvgPool3DGradNDHWCTiling, 3);

} // namespace optiling
