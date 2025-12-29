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
 * \file max_pool_grad_with_argmax_v3_nhwc_tiling.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_grad_with_argmax_v3_nhwc_tiling.h"

namespace optiling {
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NHWC = 200;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NHWC = 201;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;

void MaxPoolGradWithArgmaxV3NHWCTiling::InitializationVars()
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
    baseData.moveDataNumCacheLineT2 = CACHE_LINE_SIZE / baseData.indexBytes;

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

bool MaxPoolGradWithArgmaxV3NHWCTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }

    InitializationVars();
    return true;
}

uint64_t MaxPoolGradWithArgmaxV3NHWCTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_CHECK_RANGE_TILING_KEY_NHWC;
    if (splitData.isCheckRange == 1) {
        tilingKey = CHECK_RANGE_TILING_KEY_NHWC;
    }

    if (inputData.isInt32Meet == 0) {
        tilingKey += T3_INT64;
    }

    return tilingKey;
}

void MaxPoolGradWithArgmaxV3NHWCTiling::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData.hKernel - 1, inputData.hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData.wKernel - 1, inputData.wStride);

    int64_t inputPlaneSizeHW = hInputInner * wInputInner;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * splitData.wOutputInner;
    int64_t cOutputAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.maxDataNumInOneBlock);
    int64_t ncPlaneAlignedSize = cOutputAligned * splitData.nOutputInner;

    splitData.gradBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.inputBytes + EXTRA_BUFFER_SIZE;
    splitData.argmaxBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.indexBytes + EXTRA_BUFFER_SIZE;

    splitData.outputBufferSize = ncPlaneAlignedSize * outputPlaneSizeHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize + splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}

bool MaxPoolGradWithArgmaxV3NHWCTiling::IsMeetTargetCoreNum() const
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(inputData.nX, splitData.nOutputInner);
    int64_t tmpCOutputOuter = Ops::Base::CeilDiv(inputData.cX, splitData.cOutputInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter * tmpCOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool MaxPoolGradWithArgmaxV3NHWCTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool MaxPoolGradWithArgmaxV3NHWCTiling::TrySplitN()
{
    splitData.wOutputInner = inputData.wX;
    splitData.hOutputInner = inputData.hX;
    splitData.cOutputInner = inputData.cX;

    splitData.nOutputInner = Ops::Base::CeilDiv(inputData.nX, baseData.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.nOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = inputData.nX;
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

bool MaxPoolGradWithArgmaxV3NHWCTiling::TrySplitAlignH()
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = inputData.wX;
    splitData.cOutputInner = inputData.cX;

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

bool MaxPoolGradWithArgmaxV3NHWCTiling::TrySplitAlignW()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData.hStride;
    splitData.cOutputInner = inputData.cX;

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

bool MaxPoolGradWithArgmaxV3NHWCTiling::TrySplitAlignC()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData.hStride;
    splitData.wOutputInner = inputData.wStride;

    int64_t tmpCAligned =
        inputData.cX < baseData.moveDataNumCacheLineT2 ? inputData.cX : baseData.moveDataNumCacheLineT2;
    splitData.cOutputInner = tmpCAligned;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.cX / 2, baseData.moveDataNumCacheLineT2);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.cOutputInner = mid * baseData.moveDataNumCacheLineT2;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.cOutputInner = bestSplit * baseData.moveDataNumCacheLineT2;
        return true;
    } else {
        // hw stride 较大场景 或者 nhwc超小场景  ---> 应该对hw做更小的切分
        return false;
    }
}

void MaxPoolGradWithArgmaxV3NHWCTiling::SplitUnalignHWC()
{
    splitData.nOutputInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData.hStride;
        splitData.wOutputInner = inputData.wStride;
        int64_t tmpCAligned =
            inputData.cX < baseData.moveDataNumCacheLineT2 ? inputData.cX : baseData.moveDataNumCacheLineT2;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.wOutputInner = inputData.wX;
        splitData.hOutputInner = inputData.hX;
        splitData.cOutputInner = inputData.cX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner != 1) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    // NHW全切为1  C 超大场景 或者 NHW超小场景
    if (inputData.cX <= baseData.proDataNumInOneBeatT2) {
        return;
    } else if (IsMeetUBSize()) {
        splitData.cOutputInner = baseData.proDataNumInOneBeatT2;
        return;
    } else {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData.cX / 2, baseData.proDataNumInOneBeatT2);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.cOutputInner = mid * baseData.proDataNumInOneBeatT2;

            if (IsMeetUBSize()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData.cOutputInner = bestSplit * baseData.proDataNumInOneBeatT2;
        return;
    }
    return;
}

void MaxPoolGradWithArgmaxV3NHWCTiling::DynamicAdjustmentWH()
{
    if (splitData.hOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputOuter);
    }
}

void MaxPoolGradWithArgmaxV3NHWCTiling::SearchBestTiling()
{
    splitData.isCheckRange = 0;
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

void MaxPoolGradWithArgmaxV3NHWCTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData.wX, splitData.wOutputInner);
    int64_t tempWOutputTail = inputData.wX % splitData.wOutputInner;
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData.hX, splitData.hOutputInner);
    int64_t tempHOutputTail = inputData.hX % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData.nX, splitData.nOutputInner);
    int64_t tempNOutputTail = inputData.nX % splitData.nOutputInner;
    splitData.nOutputTail = tempNOutputTail == 0 ? splitData.nOutputInner : tempNOutputTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData.cX, splitData.cOutputInner);
    int64_t tempCOutputTail = inputData.cX % splitData.cOutputInner;
    splitData.cOutputTail = tempCOutputTail == 0 ? splitData.cOutputInner : tempCOutputTail;
}

void MaxPoolGradWithArgmaxV3NHWCTiling::DoBlockTiling()
{
    splitData.totalBaseBlockNum =
        splitData.nOutputOuter * splitData.cOutputOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

void MaxPoolGradWithArgmaxV3NHWCTiling::PrintBaseData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3NHWC", "[MaxPoolGradWithArgmaxV3NHWC] PrintBaseData start running");

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
    info << "baseData.moveDataNumCacheLineT2: " << baseData.moveDataNumCacheLineT2 << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3NHWC", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3NHWCTiling::PrintSplitData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3NHWC", "[MaxPoolGradWithArgmaxV3NHWC] PrintSplitData start running");

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
    info << "splitData.gradBufferSize: " << splitData.gradBufferSize << std::endl;
    info << "splitData.argmaxBufferSize: " << splitData.argmaxBufferSize << std::endl;
    info << "splitData.totalBufferSize: " << splitData.totalBufferSize << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3NHWC", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3NHWCTiling::SetTilingData()
{
    tilingData.set_hArgmax(inputData.hGrad);
    tilingData.set_wArgmax(inputData.wGrad);
    tilingData.set_cOutput(inputData.cX);
    tilingData.set_hOutput(inputData.hX);
    tilingData.set_wOutput(inputData.wX);
    tilingData.set_hKernel(inputData.hKernel);
    tilingData.set_wKernel(inputData.wKernel);
    tilingData.set_hStride(inputData.hStride);
    tilingData.set_wStride(inputData.wStride);
    tilingData.set_padH(inputData.hPad);
    tilingData.set_padW(inputData.wPad);
    tilingData.set_dilationH(inputData.hDilation);
    tilingData.set_dilationW(inputData.wDilation);
    tilingData.set_nOutputInner(splitData.nOutputInner);
    tilingData.set_nOutputTail(splitData.nOutputTail);
    tilingData.set_nOutputOuter(splitData.nOutputOuter);
    tilingData.set_hOutputInner(splitData.hOutputInner);
    tilingData.set_hOutputTail(splitData.hOutputTail);
    tilingData.set_hOutputOuter(splitData.hOutputOuter);
    tilingData.set_wOutputInner(splitData.wOutputInner);
    tilingData.set_wOutputTail(splitData.wOutputTail);
    tilingData.set_wOutputOuter(splitData.wOutputOuter);
    tilingData.set_cOutputInner(splitData.cOutputInner);
    tilingData.set_cOutputTail(splitData.cOutputTail);
    tilingData.set_cOutputOuter(splitData.cOutputOuter);
    tilingData.set_normalCoreProcessNum(splitData.normalCoreProcessNum);
    tilingData.set_tailCoreProcessNum(splitData.tailCoreProcessNum);
    tilingData.set_usedCoreNum(splitData.usedCoreNum);
    tilingData.set_outputBufferSize(splitData.outputBufferSize);
    tilingData.set_gradBufferSize(splitData.gradBufferSize);
    tilingData.set_argmaxBufferSize(splitData.argmaxBufferSize);
    tilingData.set_hProBatchSize(baseData.hProBatchSize);
    tilingData.set_wProBatchSize(baseData.wProBatchSize);
    tilingData.set_tilingKey(GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxV3NHWCTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxV3NHWCTiling::PostTiling()
{
    context_->SetBlockDim(tilingData.get_usedCoreNum());
    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3NHWCTiling, 1);

} // namespace optiling
