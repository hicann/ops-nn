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
#include "pool_grad_tiling_split_helper.h"
#include <iostream>

namespace optiling {
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;

void MaxPoolGradWithArgmaxNHWCTilingCommon::InitializationVars(gert::TilingContext* context_,
                                                               MaxPoolGradWithArgmaxHardwareInfo* hardwareData)
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCTilingCommon::InitializationVars()");
    InitCommonBaseInfo(context_, hardwareData->ubSize, hardwareData->coreNum);
    baseData.moveDataNumCacheLineT2 = CACHE_LINE_SIZE / baseData.indexBytes;

    baseData.isPad = 0;
    if (inputData->hPad != 0 || inputData->wPad != 0) {
        baseData.isPad = 1;
    }

    InitOverlapBatchInfo(inputData->hKernel, inputData->wKernel, inputData->hStride, inputData->wStride);
}
void MaxPoolGradWithArgmaxNHWCTilingCommon::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);

    int64_t inputPlaneSizeHW = hInputInner * wInputInner;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * splitData.wOutputInner;
    int64_t cOutputAligned = Ops::Base::CeilAlign(splitData.cOutputInner, baseData.maxDataNumInOneBlock);
    int64_t ncPlaneAlignedSize = cOutputAligned * splitData.nOutputInner;

    splitData.gradBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.inputBytes + EXTRA_BUFFER_SIZE;
    splitData.argmaxBufferSize = ncPlaneAlignedSize * inputPlaneSizeHW * baseData.indexBytes + EXTRA_BUFFER_SIZE;

    splitData.outputBufferSize = ncPlaneAlignedSize * outputPlaneSizeHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize + splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
    PrintSplitData();
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxNHWCTilingCommon::DoBufferCalculate() %d %d %d %d",
            inputData->hKernel, inputData->hStride, inputData->wKernel, inputData->wStride);
}
bool MaxPoolGradWithArgmaxNHWCTilingCommon::IsMeetTargetCoreNum() const
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    return PoolGradTiling::IsMeetTargetCoreNumNhwc(splitData, dims, baseData.coreUsedForBestPerformance);
}
bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitN()
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    return PoolGradTiling::TrySplitN(splitData, dims, baseData.coreUsedForBestPerformance, *this);
}
bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignH()
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    return PoolGradTiling::TrySplitAlignH(splitData, dims, *this);
}
bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignW()
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    return PoolGradTiling::TrySplitAlignW(splitData, dims, *this);
}
bool MaxPoolGradWithArgmaxNHWCTilingCommon::TrySplitAlignC()
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    return PoolGradTiling::TrySplitAlignC(splitData, dims, baseData.moveDataNumCacheLineT2, *this);
}
void MaxPoolGradWithArgmaxNHWCTilingCommon::SplitUnalignHWC()
{
    PoolGradTiling::PoolGradNhwcDims dims = GetNhwcDims();
    PoolGradTiling::SplitUnalignHwc(splitData, dims, baseData.isPad, baseData.isOverlap,
                                    baseData.moveDataNumCacheLineT2, baseData.proDataNumInOneBeatT2, *this);
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
    PoolGradTiling::CalcAxisOuterTail(inputData->wX, splitData.wOutputInner, splitData.wOutputOuter,
                                      splitData.wOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->hX, splitData.hOutputInner, splitData.hOutputOuter,
                                      splitData.hOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->nX, splitData.nOutputInner, splitData.nOutputOuter,
                                      splitData.nOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->cX, splitData.cOutputInner, splitData.cOutputOuter,
                                      splitData.cOutputTail);
}
void MaxPoolGradWithArgmaxNHWCTilingCommon::DoBlockTiling()
{
    PoolGradTiling::DoBlockTilingNhwc(splitData, baseData.totalCoreNum);
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

    OP_LOGI("MaxPoolGradWithArgmaxNHWCCommon", "%s", info.str().c_str());
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

    OP_LOGI("MaxPoolGradWithArgmaxNHWCCommon", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxNHWCTilingCommon::SetTilingData(gert::TilingContext* context, uint64_t key)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingData = context->GetTilingData<
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData>();
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
    tilingData->dilationH = inputData->hDilation;
    tilingData->dilationW = inputData->wDilation;
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

ge::graphStatus MaxPoolGradWithArgmaxNHWCTilingCommon::PostTiling(gert::TilingContext* context_)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData* tilingData = context_->GetTilingData<
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNHWCTilingCommonData>();
    context_->SetBlockDim(tilingData->usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

MaxPoolGradWithArgmaxNHWCSplitInfo MaxPoolGradWithArgmaxNHWCTilingCommon::GetSplitData() { return splitData; }

MaxPoolGradWithArgmaxNHWCBaseInfo MaxPoolGradWithArgmaxNHWCTilingCommon::GetBaseData() { return baseData; }

PoolGradTiling::PoolGradNhwcDims MaxPoolGradWithArgmaxNHWCTilingCommon::GetNhwcDims() const
{
    return PoolGradTiling::PoolGradNhwcDims{inputData->nX, inputData->cX,      inputData->hX,
                                            inputData->wX, inputData->hStride, inputData->wStride};
}

bool MaxPoolGradWithArgmaxNHWCTilingCommon::CheckUBSize()
{
    // ub is not enough
    splitData.nOutputInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = 1;
    splitData.cOutputInner = std::min(inputData->cX, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}
} // namespace optiling
