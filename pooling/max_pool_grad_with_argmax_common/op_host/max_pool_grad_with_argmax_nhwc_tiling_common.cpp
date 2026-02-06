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
 * \file max_pool_grad_with_argmax_nhwc_tiling_common.cpp
 * \brief
 */
#include "op_common/op_host/util/platform_util.h"
#include "max_pool_grad_with_argmax_nhwc_tiling_common.h"
#include <iostream>

namespace optiling
{
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 10240;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;

void MaxPoolGradWithArgmaxNHWCTilingCommon::InitializationVars(gert::TilingContext* context_, MaxPoolGradWithArgmaxHardwareInfo* hardwareData)
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCTilingCommon::InitializationVars()");
    baseData.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData.inputBytes = inputData->inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData.indexBytes = inputData->indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData.availableUb = hardwareData->ubSize - UB_RESVERVED_SIZE;
    baseData.totalCoreNum = hardwareData->coreNum;
    baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

    int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;
    int64_t oneBlockNumT2 = baseData.ubBlockSize / baseData.indexBytes;

    baseData.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);

    baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT2;
    baseData.moveDataNumCacheLineT2 = CACHE_LINE_SIZE / baseData.indexBytes;

    baseData.isPad = 0;
    if (inputData->hPad != 0 || inputData->wPad != 0 ) {
        baseData.isPad = 1;
    }

    baseData.hProBatchSize = 1;
    if (inputData->hKernel > inputData->hStride) {
        baseData.hProBatchSize = Ops::Base::CeilDiv(inputData->hKernel, inputData->hStride);
    }

    baseData.wProBatchSize = 1;
    if (inputData->wKernel > inputData->wStride) {
        baseData.wProBatchSize = Ops::Base::CeilDiv(inputData->wKernel, inputData->wStride);
    }

    baseData.isOverlap = 0;
    if (baseData.wProBatchSize != 1 || baseData.hProBatchSize != 1) {
        baseData.isOverlap = 1;
    }
}
void MaxPoolGradWithArgmaxNHWCTilingCommon::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);// (1 + 132 -1 )/1 = 132
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);// 1+ 45 -1 /2 = 23

    int64_t inputPlaneSizeHW = hInputInner * wInputInner; // 132*22=3036
    int64_t outputPlaneSizeHW = splitData.hOutputInner * splitData.wOutputInner;// 1
    int64_t cOutputAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.maxDataNumInOneBlock);// 6 -- 16 =  16
    int64_t ncPlaneAlignedSize = cOutputAligned * splitData.nOutputInner;//  16*1=16

    splitData.gradBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.inputBytes + EXTRA_BUFFER_SIZE;// 16*3036*2 +256=97408
    splitData.argmaxBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.indexBytes + EXTRA_BUFFER_SIZE;// 16*3036*4+256=194560

    splitData.outputBufferSize = ncPlaneAlignedSize * outputPlaneSizeHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize + splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
    PrintSplitData();
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCTilingCommon::DoBufferCalculate() %d %d %d %d", 
        inputData->hKernel, inputData->hStride, inputData->wKernel, inputData->wStride);
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::IsMeetTargetCoreNum() const
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(inputData->nX, splitData.nOutputInner);
    int64_t tmpCOutputOuter = Ops::Base::CeilDiv(inputData->cX, splitData.cOutputInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter * tmpCOutputOuter >= baseData.coreUsedForBestPerformance;
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitN()
{
    splitData.wOutputInner = inputData->wX;
    splitData.hOutputInner = inputData->hX;
    splitData.cOutputInner = inputData->cX;

    splitData.nOutputInner = Ops::Base::CeilDiv(inputData->nX, baseData.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.nOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = inputData->nX;
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

bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignH()
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = inputData->wX;
    splitData.cOutputInner = inputData->cX;

    splitData.hOutputInner = inputData->hStride;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->hX / 2, inputData->hStride);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.hOutputInner = mid * inputData->hStride;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.hOutputInner = bestSplit * inputData->hStride;
        return true;
    } else {
        return false;
    }
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignW()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData->hStride;
    splitData.cOutputInner = inputData->cX;

    splitData.wOutputInner = inputData->wStride;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->wX / 2, inputData->wStride);
        int64_t bestSplit = 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData.wOutputInner = mid * inputData->wStride;

            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        splitData.wOutputInner = bestSplit * inputData->wStride;
        return true;
    } else {
        return false;
    }
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignC()
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = inputData->hStride;
    splitData.wOutputInner = inputData->wStride;

    int64_t tmpCAligned =
        inputData->cX < baseData.moveDataNumCacheLineT2 ? inputData->cX : baseData.moveDataNumCacheLineT2;
    splitData.cOutputInner = tmpCAligned;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->cX / 2, baseData.moveDataNumCacheLineT2);
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

void MaxPoolGradWithArgmaxNHWCTilingCommon::SplitUnalignHWC()
{
    splitData.nOutputInner = 1;
    if (baseData.isPad == 0 && baseData.isOverlap == 0) {
        splitData.hOutputInner = inputData->hStride;
        splitData.wOutputInner = inputData->wStride;
        int64_t tmpCAligned =
            inputData->cX < baseData.moveDataNumCacheLineT2 ? inputData->cX : baseData.moveDataNumCacheLineT2;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.wOutputInner = inputData->wX;
        splitData.hOutputInner = inputData->hX;
        splitData.cOutputInner = inputData->cX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner != 1) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    // NHW全切为1  C 超大场景 或者 NHW超小场景
    if (inputData->cX <= baseData.proDataNumInOneBeatT2) {
        return;
    } else if (IsMeetUBSize()) {
        splitData.cOutputInner = baseData.proDataNumInOneBeatT2;
        return;
    } else {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData->cX / 2, baseData.proDataNumInOneBeatT2);
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

void MaxPoolGradWithArgmaxNHWCTilingCommon::DynamicAdjustmentWH()
{
    if (splitData.hOutputInner == 1) {
        splitData.wOutputOuter++;
        splitData.wOutputInner = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputOuter);
    } else {
        splitData.hOutputOuter++;
        splitData.hOutputInner = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputOuter);
    }
}

void MaxPoolGradWithArgmaxNHWCTilingCommon::SearchBestTiling()
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

void MaxPoolGradWithArgmaxNHWCTilingCommon::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData.wOutputOuter = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputInner);
    int64_t tempWOutputTail = inputData->wX % splitData.wOutputInner;
    splitData.wOutputTail = tempWOutputTail == 0 ? splitData.wOutputInner : tempWOutputTail;

    splitData.hOutputOuter = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputInner);
    int64_t tempHOutputTail = inputData->hX % splitData.hOutputInner;
    splitData.hOutputTail = tempHOutputTail == 0 ? splitData.hOutputInner : tempHOutputTail;

    splitData.nOutputOuter = Ops::Base::CeilDiv(inputData->nX, splitData.nOutputInner);
    int64_t tempNOutputTail = inputData->nX % splitData.nOutputInner;
    splitData.nOutputTail = tempNOutputTail == 0 ? splitData.nOutputInner : tempNOutputTail;

    splitData.cOutputOuter = Ops::Base::CeilDiv(inputData->cX, splitData.cOutputInner);
    int64_t tempCOutputTail = inputData->cX % splitData.cOutputInner;
    splitData.cOutputTail = tempCOutputTail == 0 ? splitData.cOutputInner : tempCOutputTail;
}

void MaxPoolGradWithArgmaxNHWCTilingCommon::DoBlockTiling()
{
    splitData.totalBaseBlockNum =
        splitData.nOutputOuter * splitData.cOutputOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum =
        splitData.totalBaseBlockNum - splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}


void MaxPoolGradWithArgmaxNHWCTilingCommon::PrintBaseData() const
{
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

    OP_LOGI("MaxPoolGradWithArgmaxNHWC", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxNHWCTilingCommon::PrintSplitData() const
{
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

    OP_LOGI("MaxPoolGradWithArgmaxNHWC", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxNHWCTilingCommon::SetTilingData(gert::TilingContext* context, uint64_t key)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingData =
        context->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData>();
    tilingData->hArgmax = inputData->hGrad;
    tilingData->wArgmax = inputData->wGrad;
    tilingData->cOutput = inputData->cX;
    tilingData->hOutput = inputData->hX;
    tilingData->wOutput = inputData->wX;
    tilingData->hKernel = inputData->hKernel;
    tilingData->wKernel = inputData->wKernel;
    tilingData->hStride = inputData->hStride;
    tilingData->wStride = inputData->wStride;
    tilingData->padH = inputData->hPad;
    tilingData->padW = inputData->wPad;
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
    tilingData->gradBufferSize = splitData.gradBufferSize;
    tilingData->argmaxBufferSize = splitData.argmaxBufferSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->tilingKey = key;
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCTilingCommon::DoOpTiling(gert::TilingContext* context, uint64_t key)
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData(context, key);
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxNHWCTilingCommon::PostTiling(gert::TilingContext* context_)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData>();
    context_->SetBlockDim(tilingData->usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

MaxPoolGradWithArgmaxNHWCSplitInfo MaxPoolGradWithArgmaxNHWCTilingCommon::GetSplitData()
{
    return splitData;
}

MaxPoolGradWithArgmaxNHWCBaseInfo MaxPoolGradWithArgmaxNHWCTilingCommon::GetBaseData() {
    return baseData;
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::CheckUBSize() {
    //ub is not enough
    splitData.nOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = 1;
    splitData.cOutputInner = std::min(inputData->cX, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}
}  // namespace optiling