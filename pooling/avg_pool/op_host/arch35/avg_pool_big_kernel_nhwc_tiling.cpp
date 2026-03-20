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
 * \file avg_pool_big_kernel_nhwc_tiling.cpp
 * \brief big kernel imply for pool_3d ndhwc format
 */

#include "log/log.h"
#include "util/math_util.h"
#include "platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "avg_pool_big_kernel_nhwc_tiling.h"
#include "pooling/avg_pool_v2/op_host/arch35/avg_pool_v2_common_tiling.h"

namespace optiling {

using namespace AscendC;

static constexpr uint64_t AVG_POOL_TILING_KEY_BIG_KERNEL_NHWC = 411110;

static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t BYTE_NUM_TWO = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t NO_SPLIT_KERNEL = 0;
static constexpr int64_t SPLIT_KERNEL_H = 1;
static constexpr int64_t SPLIT_KERNEL_W = 2;
static constexpr int64_t SPLIT_C = 3;
static constexpr int64_t MIN_OUT_BUFFER_LEN = 8192;
static constexpr int64_t MIN_KERNEL = 256;

bool AvgPoolCommonNHWCBigKernelTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NHWC) {
        return false;
    }
    if (inputData_.batches * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] 
            < static_cast<int64_t>(totalCoreNum_ * DIGIT_TWO)) {
        return true;
    }
    if (inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] *
        inputData_.channels * inputData_.dtypeSize < MIN_KERNEL) {
        return false;
    }
    return true;
}

ge::graphStatus AvgPoolCommonNHWCBigKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNHWCBigKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

uint64_t AvgPoolCommonNHWCBigKernelTiling::GetTilingKey() const
{
    return AVG_POOL_TILING_KEY_BIG_KERNEL_NHWC;
}
 
void AvgPoolCommonNHWCBigKernelTiling::DoUBTiling()
{
    totalIdx_ = inputData_.batches * inputData_.outShape[H_DIM] *
        inputData_.outShape[W_DIM];
    // coreNum已在tiling_base中校验过非0
    blockFactor_ = totalIdx_ / static_cast<int64_t>(totalCoreNum_);
    blockTail_ = totalIdx_ % static_cast<int64_t>(totalCoreNum_);
    coreNums_ = blockFactor_ == 0 ? totalIdx_ : static_cast<int64_t>(totalCoreNum_);
    isSigOut_ = (inputData_.outShape[H_DIM] == 1 && inputData_.outShape[W_DIM] == 1) ? 1 : 0;

    int64_t vRegSize = platform::GetVRegSize(context_) / inputData_.dtypeSize;
    int64_t blockSize = platform::GetUbBlockSize(context_) / inputData_.dtypeSize;
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) / inputData_.dtypeSize / BUFFER_NUM - vRegSize;
    int64_t oneOutChannel = std::min(MIN_OUT_BUFFER_LEN / inputData_.dtypeSize, ubAvailable / DIGIT_TWO);
    int64_t channelAlign = Ops::Base::CeilAlign(inputData_.channels, blockSize);
    int64_t oneOutChannelAlign = Ops::Base::CeilAlign(oneOutChannel, vRegSize);
    if (channelAlign > oneOutChannelAlign) {
        oneOutChannelAlign = Ops::Base::CeilAlign(channelAlign, vRegSize);
    }
    const int64_t sumBufferSize = (inputData_.dtypeSize == BYTE_NUM_TWO) ?
        oneOutChannelAlign * DIGIT_TWO : 0;
    if ((inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * channelAlign +
        oneOutChannelAlign + sumBufferSize) <= ubAvailable) {
        inUbSize_ = inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * channelAlign;
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = NO_SPLIT_KERNEL;
    } else if ((inputData_.kernelSize[W_DIM] * channelAlign + oneOutChannelAlign + sumBufferSize) <= ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign - sumBufferSize, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_H;
    } else if ((channelAlign + oneOutChannelAlign + sumBufferSize) < ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign - sumBufferSize, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_W;
    } else {
        const int64_t bufferNum = (inputData_.dtypeSize == BYTE_NUM_TWO) ? DIGIT_FOUR : DIGIT_TWO;
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable / bufferNum, vRegSize);
        outUbSize_ = Ops::Base::FloorAlign(ubAvailable / bufferNum, vRegSize);
        tilingMode_ = SPLIT_C;
    }
    onceOutNum_ = Ops::Base::FloorDiv(outUbSize_, channelAlign);
    outUbSize_ = outUbSize_ + vRegSize;
}

void AvgPoolCommonNHWCBigKernelTiling::SetTilingData()
{
    AvgPool::AvgPoolBigKernelNhwcTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolBigKernelNhwcTilingData>();
    tilingData->hInDim = inputData_.inputShape[H_DIM];
    tilingData->wInDim = inputData_.inputShape[W_DIM];
    tilingData->hOutDim = inputData_.outShape[H_DIM];
    tilingData->wOutDim = inputData_.outShape[W_DIM];
    tilingData->kH = inputData_.kernelSize[H_DIM];
    tilingData->kW = inputData_.kernelSize[W_DIM];
    tilingData->sH = inputData_.stride[H_DIM];
    tilingData->sW = inputData_.stride[W_DIM];
    tilingData->tPad = inputData_.pad[TOP_PAD_INDEX];
    tilingData->bPad = inputData_.pad[BOTTOM_PAD_INDEX];
    tilingData->lPad = inputData_.pad[LEFT_PAD_INDEX];
    tilingData->rPad = inputData_.pad[RIGHT_PAD_INDEX];
    tilingData->divisorOverride = inputData_.divisorOverride;
    tilingData->countIncludePad = static_cast<int64_t>(inputData_.countIncludePad);
    tilingData->channel = inputData_.channels;
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->totalIdx = totalIdx_;
    tilingData->coreNums = coreNums_;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->isSigOut = isSigOut_;
    tilingData->tilingMode = tilingMode_;
    tilingData->onceOutNum = onceOutNum_;
}

ge::graphStatus AvgPoolCommonNHWCBigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonNHWCBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    return ge::GRAPH_SUCCESS;
}

void AvgPoolCommonNHWCBigKernelTiling::DumpTilingInfo()
{
    AvgPool::AvgPoolBigKernelNhwcTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolBigKernelNhwcTilingData>();
    std::ostringstream str;
    str << " hInDim:" << tilingData->hInDim;
    str << " wInDim:" << tilingData->wInDim;
    str << " hOutDim:" << tilingData->hOutDim;
    str << " wOutDim:" << tilingData->wOutDim;
    str << " kH:" << tilingData->kH;
    str << " kW:" << tilingData->kW;
    str << " sH:" << tilingData->sH;
    str << " sW:" << tilingData->sW;
    str << " tPad:" << tilingData->tPad;
    str << " bPad:" << tilingData->bPad;
    str << " lPad:" << tilingData->lPad;
    str << " rPad:" << tilingData->rPad;
    str << " divisorOverride:" << tilingData->divisorOverride;
    str << " countIncludePad:" << tilingData->countIncludePad;
    str << " channel:" << tilingData->channel;
    str << " blockFactor:" << tilingData->blockFactor;
    str << " blockTail:" << tilingData->blockTail;
    str << " totalIdx:" << tilingData->totalIdx;
    str << " coreNums:" << tilingData->coreNums;
    str << " inUbSize:" << tilingData->inUbSize;
    str << " outUbSize:" << tilingData->outUbSize;
    str << " isSigOut:" << tilingData->isSigOut;
    str << " tilingMode:" << tilingData->tilingMode;
    str << " onceOutNum:" << tilingData->onceOutNum;
    OP_LOGD(context_, "Tiling inf is: %s", str.str().c_str());
}

//////////////////////////////// AvgPoolNHWCBigKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolNHWCBigKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolPlatformInfo(context_, ubSize_, totalCoreNum_);
}

ge::graphStatus AvgPoolNHWCBigKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolShapeAttrsInfo(context_, inputData_);
}

//////////////////////////////// AvgPoolV2NHWCBigKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolV2NHWCBigKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolV2PlatformInfo(context_, ubSize_, totalCoreNum_);
}

ge::graphStatus AvgPoolV2NHWCBigKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolV2ShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("AvgPool", AvgPoolNHWCBigKernelTiling, 4);
REGISTER_TILING_TEMPLATE("AvgPoolV2", AvgPoolV2NHWCBigKernelTiling, 4);

}  // namespace optiling
