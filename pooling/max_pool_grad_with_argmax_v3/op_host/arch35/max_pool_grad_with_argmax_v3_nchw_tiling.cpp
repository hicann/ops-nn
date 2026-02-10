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
 * \file max_pool_grad_with_argmax_v3_nchw_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_grad_with_argmax_v3_nchw_tiling.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NCHW = 100;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NCHW = 101;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t DOUBLE_BUFFER = 2;

void MaxPoolGradWithArgmaxV3NCHWTiling::InitializationVars()
{
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.indexBytes = inputData.indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData.availableUb = hardwareData.ubSize - UB_RESVERVED_SIZE;
    baseData.totalCoreNum = hardwareData.coreNum;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;
    int64_t oneBlockNumT2 = baseData.ubBlockSize / baseData.indexBytes;

    baseData.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);

    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT2;
    baseData.inputNCSize = inputData.nX * inputData.cX;

    baseData.isPad = 0;
    if (inputData.hPad != 0 || inputData.wPad != 0) {
        baseData.isPad = 1;
    }

    baseData.hProBatchSize = 1;
    if (inputData.hKernel > inputData.hStride) {
        baseData.hProBatchSize = Ops::Base::CeilDiv(inputData.hKernel, inputData.hStride);
    }

    baseData.wProBatchSize = 1;
    if (inputData.wKernel > inputData.wStride) {
        baseData.wProBatchSize = Ops::Base::CeilDiv(inputData.wKernel, inputData.wStride);
    }

    baseData.isOverlap = 0;
    if (baseData.wProBatchSize != 1 || baseData.hProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}

bool MaxPoolGradWithArgmaxV3NCHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }

    InitializationVars();
    // all the h and w is overlapped.
    if (baseData.hProBatchSize >= inputData.hGrad && baseData.wProBatchSize >= inputData.wGrad) {
        return false;
    }
    // ub is not enough
    splitData.highAxisInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = std::min(inputData.wX, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

uint64_t MaxPoolGradWithArgmaxV3NCHWTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NCHW;
    if (splitData.isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NCHW;
    }
    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }

    return tilingKey;
}

void MaxPoolGradWithArgmaxV3NCHWTiling::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData.hKernel - 1, inputData.hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData.wKernel - 1, inputData.wStride);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeHW = hInputInner * wInputInnerAligned;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.inputBytes;
    splitData.argmaxBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.indexBytes;
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * FLOAT32_SIZE; // 累加需要提高精度

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize + splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool MaxPoolGradWithArgmaxV3NCHWTiling::IsMeetTargetCoreNum() const
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool MaxPoolGradWithArgmaxV3NCHWTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool MaxPoolGradWithArgmaxV3NCHWTiling::TrySplitNC()
{
    splitData.wOutputInner = inputData.wX;
    splitData.hOutputInner = inputData.hX;

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

bool MaxPoolGradWithArgmaxV3NCHWTiling::TrySplitAlignH()
{
    splitData.highAxisInner = 1;
    splitData.wOutputInner = inputData.wX;

    splitData.hOutputInner = inputData.hStride;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.hX / 2, inputData.hStride);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.hOutputInner = mid * inputData.hStride;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.hOutputInner = bestSplit * inputData.hStride;
        return true;
    } else {
        return false;
    }
}

bool MaxPoolGradWithArgmaxV3NCHWTiling::TrySplitAlignW()
{
    splitData.highAxisInner = 1;
    splitData.hOutputInner = inputData.hStride;

    splitData.wOutputInner = inputData.wStride;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.wX / 2, inputData.wStride);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.wOutputInner = mid * inputData.wStride;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.wOutputInner = bestSplit * inputData.wStride;
        return true;
    } else {
        return false;
    }
}

void MaxPoolGradWithArgmaxV3NCHWTiling::SplitUnalignHW()
{
    splitData.highAxisInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData.hStride;
        splitData.wOutputInner = inputData.wStride;
    } else {
        splitData.hOutputInner = inputData.hX;
        splitData.wOutputInner = inputData.wX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner > baseData.proDataNumInOneBeatT2) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(inputData.wX, baseData.proDataNumInOneBeatT2);
    return;
}

void MaxPoolGradWithArgmaxV3NCHWTiling::DynamicAdjustmentWH()
{
    if (splitData.hOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputOuter);
    }
}

void MaxPoolGradWithArgmaxV3NCHWTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
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

void MaxPoolGradWithArgmaxV3NCHWTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    int64_t tempWOutputTail = inputData.wX % splitData.wOutputInner;
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);
    int64_t tempHOutputTail = inputData.hX % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.highAxisOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);
    int64_t tempHighAxisTail = baseData.inputNCSize % splitData.highAxisInner;
    splitData.highAxisTail = tempHighAxisTail == 0 ? splitData.highAxisInner : tempHighAxisTail;
}

void MaxPoolGradWithArgmaxV3NCHWTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void MaxPoolGradWithArgmaxV3NCHWTiling::PrintBaseData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3NCHW", "[MaxPoolGradWithArgmaxV3NCHW] PrintBaseData start running");

    std::ostringstream info;
    info << "baseData.vRegSize: " << baseData.vRegSize << std::endl;
    info << "baseData.ubBlockSize: " << baseData.ubBlockSize << std::endl;
    info << "baseData.inputBytes: " << baseData.inputBytes << std::endl;
    info << "baseData.indexBytes: " << baseData.indexBytes << std::endl;
    info << "baseData.availableUb: " << baseData.availableUb << std::endl;
    info << "baseData.maxDataNumInOneBlock: " << baseData.maxDataNumInOneBlock << std::endl;
    info << "baseData.proDataNumInOneBeatT2: " << baseData.proDataNumInOneBeatT2 << std::endl;
    info << "baseData.totalCoreNum: " << baseData.totalCoreNum << std::endl;
    info << "baseData.coreUsedForBestPerformance: " << baseData.coreUsedForBestPerformance << std::endl;
    info << "baseData.isPad: " << baseData.isPad << std::endl;
    info << "baseData.isOverlap: " << baseData.isOverlap << std::endl;
    info << "baseData.hProBatchSize: " << baseData.hProBatchSize << std::endl;
    info << "baseData.wProBatchSize: " << baseData.wProBatchSize << std::endl;
    info << "baseData.inputNCSize: " << baseData.inputNCSize << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3NCHW", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3NCHWTiling::PrintSplitData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3NCHW", "[MaxPoolGradWithArgmaxV3NCHW] PrintSplitData start running");

    std::ostringstream info;
    info << "splitData.isCheckRange: " << splitData.isCheckRange << std::endl;

    info << "splitData.highAxisInner: " << splitData.highAxisInner << std::endl;
    info << "splitData.highAxisTail: " << splitData.highAxisTail << std::endl;
    info << "splitData.highAxisOuter: " << splitData.highAxisOuter << std::endl;

    info << "splitData.hOutputInner: " << splitData.hOutputInner << std::endl;
    info << "splitData.hOutputTail: " << splitData.hOutputTail << std::endl;
    info << "splitData.hOutputOuter: " << splitData.hOutputOuter << std::endl;

    info << "splitData.wOutputInner: " << splitData.wOutputInner << std::endl;
    info << "splitData.wOutputTail: " << splitData.wOutputTail << std::endl;
    info << "splitData.wOutputOuter: " << splitData.wOutputOuter << std::endl;

    info << "splitData.normalCoreProcessNum: " << splitData.normalCoreProcessNum << std::endl;
    info << "splitData.tailCoreProcessNum: " << splitData.tailCoreProcessNum << std::endl;
    info << "splitData.usedCoreNum: " << splitData.usedCoreNum << std::endl;
    info << "splitData.totalBaseBlockNum: " << splitData.totalBaseBlockNum << std::endl;

    info << "splitData.outputBufferSize: " << splitData.outputBufferSize << std::endl;
    info << "splitData.gradBufferSize: " << splitData.gradBufferSize << std::endl;
    info << "splitData.argmaxBufferSize: " << splitData.argmaxBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3NCHW", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3NCHWTiling::SetTilingData()
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    tilingData->hArgmax = inputData.hGrad;
    tilingData->wArgmax = inputData.wGrad;
    tilingData->hOutput = inputData.hX;
    tilingData->wOutput = inputData.wX;
    tilingData->hKernel = inputData.hKernel;
    tilingData->wKernel = inputData.wKernel;
    tilingData->hStride = inputData.hStride;
    tilingData->wStride = inputData.wStride;
    tilingData->padH = inputData.hPad;
    tilingData->padW = inputData.wPad;
    tilingData->dilationH = inputData.hDilation;
    tilingData->dilationW = inputData.wDilation;
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
    tilingData->argmaxBufferSize = splitData.argmaxBufferSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->tilingKey = GetTilingKey();
}

ge::graphStatus MaxPoolGradWithArgmaxV3NCHWTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxV3NCHWTiling::PostTiling()
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3NCHWTiling, 2);

} // namespace optiling
