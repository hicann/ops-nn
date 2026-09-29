
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
 * \file max_pool3d_grad_ncdhw_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool3d_grad_tiling.h"

namespace optiling {
static constexpr int64_t KSIZE_THRESHOLD_BIG = 128;
static constexpr int64_t KSIZE_THRESHOLD = 4096;
static constexpr int64_t BIG_MERGE_BUF_ALIGN = 32;
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t KERNEL_OFFSET = 1;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

void MaxPool3DGradNCDHWTilingHelper::DoBufferCalculate()
{
    Pool3DGradNCDHWSplitInfo& splitData = GetSplitData();
    Pool3DGradNCDHWBaseInfo& baseData = GetBaseData();

    int64_t dInputInner = Ops::Base::CeilDiv(splitData.dOutputInner + inputData->dKernel - 1, inputData->dStride);
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);
    int64_t wInputAligned = Ops::Base::CeilAlign(
        splitData.wOutputInner + ((inputData->wKernel - KERNEL_OFFSET) * inputData->wDilation) * DOUBLE,
        baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeDHW = dInputInner * hInputInner * wInputInnerAligned;
    int64_t outputPlaneSizeDHW = splitData.dOutputInner * splitData.hOutputInner * wOutputInnerAligned;

    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeDHW * baseData.inputBytes;
    splitData.argmaxBufferSize = splitData.highAxisInner * dInputInner * hInputInner * wInputInner *
                                 (inputData->isInt32Meet ? INT64_SIZE : INT32_SIZE);
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeDHW * FLOAT32_SIZE;

    if (isBigKernel) {
        // NoSplit 单次装载上界: 每层 HW 按 32B 块对齐后乘 D 层数, 与 kernel 侧 curkD*hwAligned 判据一致
        const int64_t blockElems = baseData.ubBlockSize / baseData.inputBytes;
        const int64_t hwAlignedMax = Ops::Base::CeilAlign(inputData->hKernel * inputData->wKernel, blockElems);
        const int64_t fullLoadCount = inputData->dKernel * hwAlignedMax;
        const int64_t fullKernelBytes = fullLoadCount * baseData.inputBytes;

        // 前向可用预算: 前向/反向队列独立分配(求和记账), 需同时扣除 argmax、maxVal 与反向双缓冲
        const int64_t forwardInputAvailableBytes = std::max<int64_t>(
            0, (baseData.availableUb - splitData.argmaxBufferSize - splitData.gradBufferSize * DOUBLE_BUFFER -
                splitData.outputBufferSize * DOUBLE_BUFFER - BIG_MERGE_BUF_ALIGN) /
                   DOUBLE_BUFFER);
        const int64_t maxLoadCount = std::max<int64_t>(1, forwardInputAvailableBytes / baseData.inputBytes);

        if (fullKernelBytes <= forwardInputAvailableBytes) {
            splitData.inputBufferSize = fullKernelBytes;
        } else {
            splitData.inputBufferSize = maxLoadCount * baseData.inputBytes;
        }
        int64_t forwardSize = splitData.inputBufferSize * DOUBLE_BUFFER;
        int64_t backwardSize = splitData.gradBufferSize * DOUBLE_BUFFER + splitData.outputBufferSize * DOUBLE_BUFFER;
        splitData.totalBufferSize = splitData.argmaxBufferSize + forwardSize + backwardSize + BIG_MERGE_BUF_ALIGN;
    } else {
        splitData.inputBufferSize = splitData.highAxisInner *
                                    (splitData.dOutputInner +
                                     ((inputData->dKernel - KERNEL_OFFSET) * inputData->dDilation) * DOUBLE) *
                                    (splitData.hOutputInner +
                                     ((inputData->hKernel - KERNEL_OFFSET) * inputData->hDilation) * DOUBLE) *
                                    wInputAligned * baseData.inputBytes;

        int64_t forwardSize = splitData.inputBufferSize * DOUBLE_BUFFER;
        int64_t backwardSize = splitData.gradBufferSize * DOUBLE_BUFFER + splitData.outputBufferSize * DOUBLE_BUFFER;
        splitData.totalBufferSize = splitData.argmaxBufferSize + forwardSize + backwardSize;
        if (baseData.isPad == 1) {
            splitData.totalBufferSize += splitData.inputBufferSize;
        }
    }
}

bool MaxPool3DGradNCDHWTilingHelper::IsMeetUBSize()
{
    DoBufferCalculate();
    Pool3DGradNCDHWSplitInfo& splitData = GetSplitData();
    Pool3DGradNCDHWBaseInfo& baseData = GetBaseData();
    if (baseData.inputBytes == FLOAT16_SIZE) {
        return splitData.totalBufferSize <= baseData.availableUb &&
               splitData.gradBufferSize <= MAX_INPUT_ELEMENTS * baseData.inputBytes;
    }
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool MaxPool3DGradNCDHWTiling::IsCapable()
{
    base->InitializationVars(context_, ubSize_, coreNum_);
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        OP_LOGW("IsCapable", "inputFormat invalid");
        return false;
    }
    if (inputData.dDilation != 1 || inputData.hDilation != 1 || inputData.wDilation != 1) {
        OP_LOGI("IsCapable", "dDilation:%ld,  hDilation:%ld,  wDilation:%ld", inputData.dDilation, inputData.hDilation,
                inputData.wDilation);
        return false;
    }
    int64_t inputDataCount = inputData.nX * inputData.cX * inputData.dX * inputData.hX * inputData.wX;
    if (inputDataCount > MAX_INT32) {
        OP_LOGI("IsCapable", "inputDataCount:%ld exceeds int32, fast div not supported", inputDataCount);
        return false;
    }
    int64_t kv = inputData.dKernel * inputData.hKernel * inputData.wKernel;
    // 判断大小 kernel
    base->isBigKernel = (kv >= KSIZE_THRESHOLD_BIG) ? 1 : 0;
    base->GetSplitData().isBigKernel = base->isBigKernel;

    bool ksizeCheck = true;
    if (base->GetBaseData().dProBatchSize == 1 && base->GetBaseData().hProBatchSize == 1 &&
        base->GetBaseData().wProBatchSize == 1) {
        ksizeCheck = kv < KSIZE_THRESHOLD;
    }

    // ub is not enough
    base->GetSplitData().highAxisInner = 1;
    base->GetSplitData().dOutputInner = 1;
    base->GetSplitData().hOutputInner = 1;
    base->GetSplitData().wOutputInner = std::min(inputData.wX, base->GetBaseData().proDataNumInOneBeatT2);
    base->DoBufferCalculate();

    OP_LOGI("IsCapable", "kv=%ld, isBigKernel:%ld, totalBufferSize:%ld, availableUb:%ld", kv, base->isBigKernel,
            base->GetSplitData().totalBufferSize, base->GetBaseData().availableUb);
    return ksizeCheck && base->GetSplitData().totalBufferSize <= base->GetBaseData().availableUb;
}

uint64_t MaxPool3DGradNCDHWTiling::GetTilingKey() const
{
    uint32_t idxDtype = TPL_INT32;
    uint32_t isSimt = 0;
    uint32_t isChannelLast = 0;
    uint32_t useINT64Index = 0;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, isCheckRange_, useINT64Index);
}

ge::graphStatus MaxPool3DGradNCDHWTiling::DoOpTiling()
{
    auto ret = base->DoOpTiling(context_);
    isCheckRange_ = base->GetSplitData().isCheckRange;
    return ret;
}

ge::graphStatus MaxPool3DGradNCDHWTiling::PostTiling() { return base->PostTiling(context_, GetTilingKey()); }

REGISTER_TILING_TEMPLATE("MaxPool3DGrad", MaxPool3DGradNCDHWTiling, 0);

} // namespace optiling
