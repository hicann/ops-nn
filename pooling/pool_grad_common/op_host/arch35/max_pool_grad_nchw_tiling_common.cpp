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
 * \file max_pool_grad_nchw_tiling_common.cpp
 * \brief NCHW格式MaxPoolGrad通用Tiling实现
 */

#include "platform/platform_info.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_grad_nchw_tiling_common.h"
#include "pool_grad_tiling_split_helper.h"

namespace optiling {
static constexpr int64_t NO_CHECK_RANGE_TILING_KEY_NCHW = 100;
static constexpr int64_t CHECK_RANGE_TILING_KEY_NCHW = 101;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t DOUBLE_BUFFER = 2;

void MaxPoolGradNCHWTilingCommon::InitializationVars(gert::TilingContext* context_,
                                                     MaxPoolGradWithArgmaxHardwareInfo* hardwareData)
{
    InitCommonBaseInfo(context_, hardwareData->ubSize, hardwareData->coreNum);
    baseData.inputNCSize = inputData->nX * inputData->cX;

    baseData.isPad = 0;
    if (inputData->hPad != 0 || inputData->wPad != 0) {
        baseData.isPad = 1;
    }

    if (((inputData->wGrad - 1) * inputData->wStride + inputData->wKernel) > inputData->wX ||
        ((inputData->hGrad - 1) * inputData->hStride + inputData->hKernel) > inputData->hX) {
        baseData.isPad = 1;
    }

    InitOverlapBatchInfo(inputData->hKernel, inputData->wKernel, inputData->hStride, inputData->wStride);
}

bool MaxPoolGradNCHWTilingCommon::CheckUBSize()
{
    // all the h and w is overlapped.
    if (baseData.hProBatchSize >= inputData->hGrad && baseData.wProBatchSize >= inputData->wGrad) {
        return false;
    }
    // ub is not enough
    splitData.highAxisInner = 1;
    splitData.hOutputInner = 1;
    splitData.wOutputInner = std::min(inputData->wX, baseData.proDataNumInOneBeatT2);
    DoBufferCalculate();
    return splitData.totalBufferSize <= baseData.availableUb;
}

void MaxPoolGradNCHWTilingCommon::DoBufferCalculate()
{
    // The calculation only involves inner.
    int64_t hInputInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(wInputInner, baseData.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

    int64_t inputPlaneSizeHW = hInputInner * wInputInnerAligned;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

    splitData.inputBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.inputBytes;
    splitData.gradBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.inputBytes;
    splitData.argmaxBufferSize = splitData.highAxisInner * inputPlaneSizeHW * baseData.indexBytes;
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * FLOAT32_SIZE;

    int64_t tmpTotalBufferSize = splitData.outputBufferSize + splitData.gradBufferSize + splitData.argmaxBufferSize;
    splitData.totalBufferSize = tmpTotalBufferSize * DOUBLE_BUFFER;
}
bool MaxPoolGradNCHWTilingCommon::IsMeetTargetCoreNum() const
{
    PoolGradTiling::PoolGradNchwDims dims = GetNchwDims();
    return PoolGradTiling::IsMeetTargetCoreNumNchw(splitData, dims, baseData.inputNCSize,
                                                   baseData.coreUsedForBestPerformance);
}
bool MaxPoolGradNCHWTilingCommon::TrySplitNC()
{
    PoolGradTiling::PoolGradNchwDims dims = GetNchwDims();
    return PoolGradTiling::TrySplitNc(splitData, dims, baseData.inputNCSize, baseData.coreUsedForBestPerformance,
                                      *this);
}
bool MaxPoolGradNCHWTilingCommon::TrySplitAlignH()
{
    PoolGradTiling::PoolGradNchwDims dims = GetNchwDims();
    return PoolGradTiling::TrySplitAlignH(splitData, dims, *this);
}
bool MaxPoolGradNCHWTilingCommon::TrySplitAlignW()
{
    PoolGradTiling::PoolGradNchwDims dims = GetNchwDims();
    return PoolGradTiling::TrySplitAlignW(splitData, dims, *this);
}
void MaxPoolGradNCHWTilingCommon::SplitUnalignHW()
{
    PoolGradTiling::PoolGradNchwDims dims = GetNchwDims();
    PoolGradTiling::SplitUnalignHw(splitData, dims, baseData.isPad, baseData.isOverlap, baseData.proDataNumInOneBeatT2,
                                   *this);
}

void MaxPoolGradNCHWTilingCommon::SearchBestTiling()
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

    splitData.isCheckRange = 1;
    SplitUnalignHW();
    return;
}

void MaxPoolGradNCHWTilingCommon::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    PoolGradTiling::CalcAxisOuterTail(inputData->wX, splitData.wOutputInner, splitData.wOutputOuter,
                                      splitData.wOutputTail);
    PoolGradTiling::CalcAxisOuterTail(inputData->hX, splitData.hOutputInner, splitData.hOutputOuter,
                                      splitData.hOutputTail);
    PoolGradTiling::CalcAxisOuterTail(baseData.inputNCSize, splitData.highAxisInner, splitData.highAxisOuter,
                                      splitData.highAxisTail);
}
void MaxPoolGradNCHWTilingCommon::DoBlockTiling()
{
    PoolGradTiling::DoBlockTilingNchw(splitData, baseData.totalCoreNum);
}
void MaxPoolGradNCHWTilingCommon::PrintBaseData() const
{
    OP_LOGD("MaxPoolGradNCHW", "[MaxPoolGradNCHW] PrintBaseData start running");

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

    OP_LOGI("MaxPoolGradNCHW", "%s", info.str().c_str());
}

void MaxPoolGradNCHWTilingCommon::PrintSplitData() const
{
    OP_LOGD("MaxPoolGradNCHW", "[MaxPoolGradNCHW] PrintSplitData start running");

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

    OP_LOGI("MaxPoolGradNCHW", "%s", info.str().c_str());
}

void MaxPoolGradNCHWTilingCommon::SetTilingData(gert::TilingContext* context, uint64_t key)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData* tilingData = context->GetTilingData<
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    tilingData->hArgmax = inputData->hGrad;
    tilingData->wArgmax = inputData->wGrad;
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
    tilingData->inputBufferSize = splitData.inputBufferSize;
    tilingData->outputBufferSize = splitData.outputBufferSize;
    tilingData->gradBufferSize = splitData.gradBufferSize;
    tilingData->argmaxBufferSize = splitData.argmaxBufferSize;
    tilingData->hProBatchSize = baseData.hProBatchSize;
    tilingData->wProBatchSize = baseData.wProBatchSize;
    tilingData->tilingKey = key;
    tilingData->isPad = baseData.isPad;
}

MaxPoolGradNCHWSplitInfo MaxPoolGradNCHWTilingCommon::GetSplitData() const { return splitData; }

MaxPoolGradNCHWBaseInfo MaxPoolGradNCHWTilingCommon::GetBaseData() { return baseData; }

PoolGradTiling::PoolGradNchwDims MaxPoolGradNCHWTilingCommon::GetNchwDims() const
{
    return PoolGradTiling::PoolGradNchwDims{inputData->hX, inputData->wX, inputData->hStride, inputData->wStride};
}

ge::graphStatus MaxPoolGradNCHWTilingCommon::PostTiling(gert::TilingContext* context_)
{
    MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData* tilingData = context_->GetTilingData<
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    context_->SetTilingKey(tilingData->tilingKey);
    context_->SetBlockDim(tilingData->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
