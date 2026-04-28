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
 * \file max_pool_grad_nchw_tiling.cpp
 * \brief NCHW format MaxPoolGrad unified tiling
 */
#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_grad_tiling.h"

using namespace PoolGradNameSpace;

namespace optiling {
static constexpr int64_t KSIZE_THRESHOLD = 128;
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DOUBLE = 2;

void MaxPoolGradNCHWTilingHelper::DoBufferCalculate()
{
    if (inputData->hKernel * inputData->wKernel < KSIZE_THRESHOLD) {
        int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
        int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);
        int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.maxDataNumInOneBlock);
        int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);
        int64_t wInputAligned = Ops::Base::CeilAlign(
            splitData.wOutputInner + ((inputData->wKernel - KERNEL_OFFSET) * inputData->wDilation) * DOUBLE,
            baseData.maxDataNumInOneBlock);

        int64_t inputPlaneSizeHW = hInputInner * wInputInnerAligned;
        int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

        splitData.inputBufferSize =
            splitData.highAxisInner *
            (splitData.hOutputInner + ((inputData->hKernel - KERNEL_OFFSET) * inputData->hDilation) * DOUBLE) *
            wInputAligned * baseData.inputBytes;
        splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.inputBytes;
        splitData.argmaxBufferSize =
            splitData.highAxisInner * inputPlaneSizeHW * (inputData->isInt32Meet ? INT64_SIZE : INT32_SIZE);
        splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * FLOAT32_SIZE;

        int64_t tmpTotalBufferSize = splitData.inputBufferSize + splitData.outputBufferSize + splitData.gradBufferSize;
        splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER + splitData.argmaxBufferSize;
        if (baseData.isPad == 1) {
            splitData.totalBufferSize += splitData.inputBufferSize;
        }
    } else {
        const int64_t hArgmaxInner = std::min<int64_t>(
            inputData->hGrad,
            Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride));
        const int64_t wArgmaxInner = std::min<int64_t>(
            inputData->wGrad,
            Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride));

        const int64_t wArgmaxInnerAligned = Ops::Base::CeilAlign(wArgmaxInner, baseData.maxDataNumInOneBlock);
        const int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

        const int64_t gradPlaneSizeHW = hArgmaxInner * wArgmaxInnerAligned;

        const int64_t argmaxPlaneSizeHW = hArgmaxInner * wArgmaxInner;

        const int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

        splitData.gradBufferSize = splitData.highAxisInner * gradPlaneSizeHW * baseData.inputBytes;
        splitData.argmaxBufferSize = splitData.highAxisInner * argmaxPlaneSizeHW * baseData.indexBytes;
        splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * BIG_FLOAT32_SIZE;

        const int64_t fullKernelCount = inputData->hKernel * inputData->wKernel;
        const int64_t fullKernelBytes = fullKernelCount * baseData.inputBytes;
        const int64_t forwardInputAvailableBytes =
            std::max<int64_t>(0, baseData.availableUb - splitData.argmaxBufferSize - BIG_MERGE_BUF_ALIGN);
        const int64_t maxLoadCount = std::max<int64_t>(1, forwardInputAvailableBytes / baseData.inputBytes);

        if (fullKernelBytes <= forwardInputAvailableBytes) {
            splitData.inputBufferSize = fullKernelBytes;
        } else {
            splitData.inputBufferSize = maxLoadCount * baseData.inputBytes;
        }

        const int64_t forwardStageBufferSize = splitData.inputBufferSize * BIG_DOUBLE_BUFFER + BIG_MERGE_BUF_ALIGN;
        const int64_t backwardStageBufferSize =
            splitData.gradBufferSize * BIG_DOUBLE_BUFFER + splitData.outputBufferSize * BIG_DOUBLE_BUFFER;

        splitData.totalBufferSize =
            splitData.argmaxBufferSize + std::max<int64_t>(forwardStageBufferSize, backwardStageBufferSize);
    }
}

bool MaxPoolGradNCHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        OP_LOGI("IsCapable", "inputFormat error, expected NCHW");
        return false;
    }
    if (inputData.hDilation != 1 || inputData.wDilation != 1) {
        OP_LOGI("IsCapable", "hDilation:%ld, wDilation:%ld", inputData.hDilation, inputData.wDilation);
        return false;
    }
    commonTiling.InitializationVars(context_, &hwInfo);
    return commonTiling.CheckUBSize() && (inputData.isInt32Meet == 0);
}

uint64_t MaxPoolGradNCHWTiling::GetTilingKey() const
{
    uint32_t indicesDtype = (inputData.hX * inputData.wX <= static_cast<int64_t>(MAX_INT32)) ? TPL_INT32 : TPL_INT64;
    uint32_t format = TPL_NCHW_FORMAT;
    uint32_t isCheckRange = commonTiling.GetSplitData().isCheckRange;
    uint32_t kernelMode =
        (inputData.hKernel * inputData.wKernel >= KSIZE_THRESHOLD) ? TPL_NCHW_BIG_KERNEL : TPL_NCHW_SMALL_KERNEL;
    return GET_TPL_TILING_KEY(kernelMode, format, indicesDtype, isCheckRange);
}

ge::graphStatus MaxPoolGradNCHWTiling::DoOpTiling()
{
    commonTiling.InitializationVars(context_, &hwInfo);
    return commonTiling.DoOpTiling(context_, GetTilingKey());
}

ge::graphStatus MaxPoolGradNCHWTiling::PostTiling()
{
    return commonTiling.PostTiling(context_);
}

ge::graphStatus MaxPoolGradNCHWTiling::GetShapeAttrsInfo()
{
    auto ret = MaxPoolGradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        OP_LOGI("GetShapeAttrsInfo", "inputFormat error, expected NCHW");
        return ge::GRAPH_PARAM_INVALID;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradNCHWTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const MaxPoolGradWithArgmaxCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        hwInfo.coreNum = compileInfoPtr->coreNum;
        hwInfo.ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        hwInfo.coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        hwInfo.ubSize = static_cast<int64_t>(ubSizePlatform);
    }

    OP_CHECK_IF(hwInfo.coreNum == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    coreNum_ = hwInfo.coreNum;
    ubSize_ = hwInfo.ubSize;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPoolGrad", MaxPoolGradNCHWTiling, 0);

} // namespace optiling