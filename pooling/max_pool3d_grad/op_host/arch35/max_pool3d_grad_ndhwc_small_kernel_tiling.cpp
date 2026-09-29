/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "max_pool3d_grad_tiling.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t KERNEL_OFFSET = 1;
static constexpr int64_t BIG_KERNEL_THRESHOLD = 128;
static constexpr int64_t BATCH_LIMIT = 256;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

void MaxPool3DGradNDHWCTilingImpl::DoBufferCalculate()
{
    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner + inputData->dKernel - 1, inputData->dStride);
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);

    int64_t cAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.maxDataNumInOneBlock);

    splitData.gradBufferSize = splitData.nOutputInner * dInputInner * hInputInner * wInputInner * cAligned *
                               baseData.inputBytes;
    splitData.outputBufferSize = splitData.nOutputInner * splitData.dOutputInner * splitData.hOutputInner *
                                 splitData.wOutputInner * cAligned * FLOAT32_SIZE;
    splitData.argmaxBufferSize = splitData.nOutputInner * splitData.cOutputInner * dInputInner * hInputInner *
                                 wInputInner * baseData.indexBytes;

    if (splitData.isBigKernel == 1) {
        const int64_t kv = inputData->dKernel * inputData->hKernel * inputData->wKernel;
        const int64_t mergeSize = cAligned * baseData.inputBytes;
        const int64_t fullWindowBytes = kv * cAligned * baseData.inputBytes;
        const int64_t minInput = cAligned * baseData.inputBytes;
        const int64_t forwardInputAvailableBytes = std::max<int64_t>(
            0, (baseData.availableUb - splitData.argmaxBufferSize - mergeSize) / DOUBLE_BUFFER);
        const int64_t maxLoadBytes = std::max<int64_t>(minInput, forwardInputAvailableBytes);
        splitData.inputBufferSize = std::min(fullWindowBytes, maxLoadBytes);
        const int64_t forwardSize = splitData.inputBufferSize * DOUBLE_BUFFER + mergeSize;
        const int64_t backwardSize = splitData.gradBufferSize + splitData.outputBufferSize;
        const int64_t sharedSize = (forwardSize > backwardSize) ? forwardSize : backwardSize;
        splitData.totalBufferSize = splitData.argmaxBufferSize + sharedSize;
        return;
    }

    int64_t dInExpect = splitData.dOutputInner +
                        (inputData->dKernel - KERNEL_OFFSET) * inputData->dDilation * DOUBLE_BUFFER;
    int64_t hInExpect = splitData.hOutputInner +
                        (inputData->hKernel - KERNEL_OFFSET) * inputData->hDilation * DOUBLE_BUFFER;
    int64_t wInExpect = splitData.wOutputInner +
                        (inputData->wKernel - KERNEL_OFFSET) * inputData->wDilation * DOUBLE_BUFFER;
    splitData.inputBufferSize = splitData.nOutputInner * dInExpect * hInExpect * wInExpect * cAligned *
                                baseData.inputBytes;

    int64_t calcSize = baseData.isPad ? splitData.inputBufferSize : 0;
    int64_t forwardSize = splitData.inputBufferSize * DOUBLE_BUFFER + calcSize;
    int64_t backwardSize = splitData.gradBufferSize * DOUBLE_BUFFER + splitData.outputBufferSize * DOUBLE_BUFFER;
    int64_t sharedSize = (forwardSize > backwardSize) ? forwardSize : backwardSize;
    splitData.totalBufferSize = splitData.argmaxBufferSize + sharedSize;
}

bool MaxPool3DGradNDHWCTilingImpl::IsMeetUBSize()
{
    DoBufferCalculate();
    if (baseData.inputBytes == FLOAT16_SIZE) {
        return splitData.totalBufferSize <= baseData.availableUb &&
               splitData.inputBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes &&
               splitData.gradBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes;
    }
    return splitData.totalBufferSize <= baseData.availableUb;
}

void MaxPool3DGradNDHWCTilingImpl::SetTilingData(gert::TilingContext* context)
{
    Pool3DGradNDHWCTilingData* tilingData = context->GetTilingData<Pool3DGradNDHWCTilingData>();

    tilingData->base.dArgmax = inputData->dGrad;
    tilingData->base.hArgmax = inputData->hGrad;
    tilingData->base.wArgmax = inputData->wGrad;
    tilingData->base.dOutput = inputData->dX;
    tilingData->base.hOutput = inputData->hX;
    tilingData->base.wOutput = inputData->wX;
    tilingData->base.dKernel = inputData->dKernel;
    tilingData->base.hKernel = inputData->hKernel;
    tilingData->base.wKernel = inputData->wKernel;
    tilingData->base.dStride = inputData->dStride;
    tilingData->base.hStride = inputData->hStride;
    tilingData->base.wStride = inputData->wStride;
    tilingData->base.padD = inputData->dPad;
    tilingData->base.padH = inputData->hPad;
    tilingData->base.padW = inputData->wPad;
    tilingData->base.padDBack = inputData->dPadBack;
    tilingData->base.padHBack = inputData->hPadBack;
    tilingData->base.padWBack = inputData->wPadBack;
    tilingData->base.dilationD = inputData->dDilation;
    tilingData->base.dilationH = inputData->hDilation;
    tilingData->base.dilationW = inputData->wDilation;

    tilingData->base.highAxisInner = splitData.nOutputInner;
    tilingData->base.highAxisTail = splitData.nOutputTail;
    tilingData->base.highAxisOuter = splitData.nOutputOuter;
    tilingData->base.dOutputInner = splitData.dOutputInner;
    tilingData->base.dOutputTail = splitData.dOutputTail;
    tilingData->base.dOutputOuter = splitData.dOutputOuter;
    tilingData->base.hOutputInner = splitData.hOutputInner;
    tilingData->base.hOutputTail = splitData.hOutputTail;
    tilingData->base.hOutputOuter = splitData.hOutputOuter;
    tilingData->base.wOutputInner = splitData.wOutputInner;
    tilingData->base.wOutputTail = splitData.wOutputTail;
    tilingData->base.wOutputOuter = splitData.wOutputOuter;
    tilingData->base.normalCoreProcessNum = splitData.normalCoreProcessNum;
    tilingData->base.tailCoreProcessNum = splitData.tailCoreProcessNum;
    tilingData->base.usedCoreNum = splitData.usedCoreNum;
    tilingData->base.inputBufferSize = splitData.inputBufferSize;
    tilingData->base.outputBufferSize = splitData.outputBufferSize;
    tilingData->base.gradBufferSize = splitData.gradBufferSize;
    tilingData->base.argmaxBufferSize = splitData.argmaxBufferSize;
    tilingData->base.dProBatchSize = baseData.dProBatchSize;
    tilingData->base.hProBatchSize = baseData.hProBatchSize;
    tilingData->base.wProBatchSize = baseData.wProBatchSize;

    tilingData->cDim = inputData->cX;
    tilingData->cOutputInner = splitData.cOutputInner;
    tilingData->cOutputTail = splitData.cOutputTail;
    tilingData->cOutputOuter = splitData.cOutputOuter;
    tilingData->isBigKernel = splitData.isBigKernel;
}

bool MaxPool3DGradNDHWCSmallKernelTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NDHWC) {
        OP_LOGW("IsCapable", "inputFormat invalid, expected NDHWC");
        return false;
    }

    if (inputData.dDilation != 1 || inputData.hDilation != 1 || inputData.wDilation != 1) {
        OP_LOGI("IsCapable", "dilation not supported: d=%ld, h=%ld, w=%ld", inputData.dDilation, inputData.hDilation,
                inputData.wDilation);
        return false;
    }

    int64_t inputDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    if (inputDataCount > INT32_MAX) {
        OP_LOGI("IsCapable", "inputDataCount:%ld exceeds int32, fast div not supported", inputDataCount);
        return false;
    }

    int64_t kv = inputData.dKernel * inputData.hKernel * inputData.wKernel;

    ndhwcBase->InitializationVars(context_, ubSize_, coreNum_);

    int64_t batchSize = ndhwcBase->GetBaseData().dProBatchSize * ndhwcBase->GetBaseData().hProBatchSize *
                        ndhwcBase->GetBaseData().wProBatchSize;
    if (batchSize >= BATCH_LIMIT) {
        OP_LOGI("IsCapable", "batch size too large: %ld", batchSize);
        return false;
    }

    auto& split = ndhwcBase->GetSplitData();
    split.isBigKernel = (kv >= BIG_KERNEL_THRESHOLD) ? 1 : 0;

    split.nOutputInner = 1;
    split.dOutputInner = 1;
    split.hOutputInner = 1;
    split.wOutputInner = (split.isBigKernel == 1) ?
                             1 :
                             std::min(inputData.wX, ndhwcBase->GetBaseData().proDataNumInOneBeat);
    split.cOutputInner = std::min(inputData.cX, ndhwcBase->GetBaseData().proDataNumInOneBeat);

    ndhwcBase->DoBufferCalculate();
    return split.totalBufferSize <= ndhwcBase->GetBaseData().availableUb;
}

uint64_t MaxPool3DGradNDHWCSmallKernelTiling::GetTilingKey() const
{
    uint32_t idxDtype = TPL_INT32;
    uint32_t isSimt = 0;
    uint32_t isChannelLast = 1;
    uint32_t useINT64Index = 0;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, static_cast<uint32_t>(isCheckRange_), useINT64Index);
}

ge::graphStatus MaxPool3DGradNDHWCSmallKernelTiling::DoOpTiling()
{
    ndhwcBase->DoOpTiling(context_);
    isCheckRange_ = ndhwcBase->GetSplitData().isCheckRange;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradNDHWCSmallKernelTiling::PostTiling()
{
    return ndhwcBase->PostTiling(context_, GetTilingKey());
}

REGISTER_TILING_TEMPLATE("MaxPool3DGrad", MaxPool3DGradNDHWCSmallKernelTiling, 2);

} // namespace optiling
