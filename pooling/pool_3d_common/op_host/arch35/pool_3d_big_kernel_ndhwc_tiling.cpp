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
 * \file pool_3d_big_kernel_ndhwc_tiling.cpp
 * \brief big kernel imply for pool_3d ndhwc format
 */

#include "op_util.h"
#include "pool_tiling_templates_registry.h"
#include "pool_3d_big_kernel_ndhwc_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"


namespace optiling {

using namespace AscendC;

static constexpr uint64_t POOL_3D_TILING_KEY_BIG_KERNEL_NDHWC = 411110;

static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t BYTE_NUM_TWO = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t NO_SPLIT_KERNEL = 0;
static constexpr int64_t SPLIT_KERNEL_D = 1;
static constexpr int64_t SPLIT_KERNEL_H = 2;
static constexpr int64_t SPLIT_KERNEL_W = 3;
static constexpr int64_t SPLIT_C = 4;
static constexpr int64_t MIN_OUT_BUFFER_LEN = 8192;
static constexpr int64_t MIN_KERNEL = 256;

bool Pool3DBigKernelNDHWCTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NDHWC) {
        return false;
    }
    if (inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] 
            < static_cast<int64_t>(totalCoreNum_ * DIGIT_TWO)) {
        return true;
    }
    if (inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] *
        inputData_.channels * inputData_.dtypeSize < MIN_KERNEL) {
        return false;
    }
    return true;
}

ge::graphStatus Pool3DBigKernelNDHWCTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DBigKernelNDHWCTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

uint64_t Pool3DBigKernelNDHWCTiling::GetTilingKey() const
{
    return POOL_3D_TILING_KEY_BIG_KERNEL_NDHWC;
}
 
void Pool3DBigKernelNDHWCTiling::DoUBTiling()
{
    totalIdx_ = inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] *
        inputData_.outShape[W_DIM];
    // coreNum已在tiling_base中校验过非0
    blockFactor_ = totalIdx_ / static_cast<int64_t>(totalCoreNum_);
    blockTail_ = totalIdx_ % static_cast<int64_t>(totalCoreNum_);
    coreNums_ = blockFactor_ == 0 ? totalIdx_ : static_cast<int64_t>(totalCoreNum_);
    isSigOut_ = (inputData_.outShape[D_DIM] == 1 && inputData_.outShape[H_DIM] == 1 &&
        inputData_.outShape[W_DIM] == 1) ? 1 : 0;

    int64_t vRegSize = Ops::Base::GetVRegSize(context_) / inputData_.dtypeSize;
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_) / inputData_.dtypeSize;
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) / inputData_.dtypeSize / BUFFER_NUM - vRegSize;
    int64_t oneOutChannel = std::min(MIN_OUT_BUFFER_LEN / inputData_.dtypeSize, ubAvailable / DIGIT_TWO);
    int64_t channelAlign = Ops::Base::FloorAlign(inputData_.channels, blockSize);
    int64_t oneOutChannelAlign = Ops::Base::FloorAlign(oneOutChannel, vRegSize);
    if (channelAlign > oneOutChannelAlign) {
        oneOutChannelAlign = Ops::Base::FloorAlign(channelAlign, vRegSize);
    }
    const int64_t sumBufferSize = (!isMaxPool3D_ && inputData_.dtypeSize == BYTE_NUM_TWO) ?
        oneOutChannelAlign * DIGIT_TWO : 0;
    if ((inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * channelAlign +
        oneOutChannelAlign + sumBufferSize) <= ubAvailable) {
        inUbSize_ = inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] *
            channelAlign;
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = NO_SPLIT_KERNEL;
    } else if ((inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * channelAlign + oneOutChannelAlign +
        sumBufferSize) <= ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign - sumBufferSize, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_D;
    } else if ((inputData_.kernelSize[W_DIM] * channelAlign + oneOutChannelAlign + sumBufferSize) <= ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign - sumBufferSize, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_H;
    } else if ((channelAlign + oneOutChannelAlign + sumBufferSize) < ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign - sumBufferSize, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_W;
    } else {
        const int64_t bufferNum = (!isMaxPool3D_ && inputData_.dtypeSize == BYTE_NUM_TWO) ? DIGIT_FOUR : DIGIT_TWO;
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable / bufferNum, vRegSize);
        outUbSize_ = Ops::Base::FloorAlign(ubAvailable / bufferNum, vRegSize);
        tilingMode_ = SPLIT_C;
    }
    onceOutNum_ = Ops::Base::FloorDiv(outUbSize_, channelAlign);
    outUbSize_ = outUbSize_ + vRegSize;
}

void Pool3DBigKernelNDHWCTiling::SetTilingData()
{
    tiling_.set_dInDim(inputData_.inputShape[D_DIM]);
    tiling_.set_hInDim(inputData_.inputShape[H_DIM]);
    tiling_.set_wInDim(inputData_.inputShape[W_DIM]);
    tiling_.set_dOutDim(inputData_.outShape[D_DIM]);
    tiling_.set_hOutDim(inputData_.outShape[H_DIM]);
    tiling_.set_wOutDim(inputData_.outShape[W_DIM]);
    tiling_.set_kD(inputData_.kernelSize[D_DIM]);
    tiling_.set_kH(inputData_.kernelSize[H_DIM]);
    tiling_.set_kW(inputData_.kernelSize[W_DIM]);
    tiling_.set_sD(inputData_.stride[D_DIM]);
    tiling_.set_sH(inputData_.stride[H_DIM]);
    tiling_.set_sW(inputData_.stride[W_DIM]);
    tiling_.set_fPad(inputData_.pad[FRONT_PAD_INDEX]);
    tiling_.set_backPad(inputData_.pad[BACKEND_PAD_INDEX]);
    tiling_.set_tPad(inputData_.pad[TOP_PAD_INDEX]);
    tiling_.set_bottomPad(inputData_.pad[BOTTOM_PAD_INDEX]);
    tiling_.set_lPad(inputData_.pad[LEFT_PAD_INDEX]);
    tiling_.set_rPad(inputData_.pad[RIGHT_PAD_INDEX]);
    tiling_.set_dD(inputData_.dilation[D_DIM]);
    tiling_.set_dH(inputData_.dilation[H_DIM]);
    tiling_.set_dW(inputData_.dilation[W_DIM]);
    tiling_.set_divisorOverride(inputData_.divisorOverride);
    tiling_.set_countIncludePad(static_cast<int64_t>(inputData_.countIncludePad));
    tiling_.set_channel(inputData_.channels);
    tiling_.set_blockFactor(blockFactor_);
    tiling_.set_blockTail(blockTail_);
    tiling_.set_totalIdx(totalIdx_);
    tiling_.set_coreNums(coreNums_);
    tiling_.set_inUbSize(inUbSize_);
    tiling_.set_outUbSize(outUbSize_);
    tiling_.set_isSigOut(isSigOut_);
    tiling_.set_tilingMode(tilingMode_);
    tiling_.set_onceOutNum(onceOutNum_);
}

ge::graphStatus Pool3DBigKernelNDHWCTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DBigKernelNDHWCTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    tiling_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DBigKernelNDHWCTiling::DumpTilingInfo()
{
    std::string str;
    str += " dInDim:" + std::to_string(tiling_.get_dInDim());
    str += " hInDim:" + std::to_string(tiling_.get_hInDim());
    str += " wInDim:" + std::to_string(tiling_.get_wInDim());
    str += " dOutDim:" + std::to_string(tiling_.get_dOutDim());
    str += " hOutDim:" + std::to_string(tiling_.get_hOutDim());
    str += " wOutDim:" + std::to_string(tiling_.get_wOutDim());
    str += " kD:" + std::to_string(tiling_.get_kD());
    str += " kH:" + std::to_string(tiling_.get_kH());
    str += " kW:" + std::to_string(tiling_.get_kW());
    str += " sD:" + std::to_string(tiling_.get_sD());
    str += " sH:" + std::to_string(tiling_.get_sH());
    str += " sW:" + std::to_string(tiling_.get_sW());
    str += " fPad:" + std::to_string(tiling_.get_fPad());
    str += " backPad:" + std::to_string(tiling_.get_backPad());
    str += " tPad:" + std::to_string(tiling_.get_tPad());
    str += " bottomPad:" + std::to_string(tiling_.get_bottomPad());
    str += " lPad:" + std::to_string(tiling_.get_lPad());
    str += " rPad:" + std::to_string(tiling_.get_rPad());
    str += " dD:" + std::to_string(tiling_.get_dD());
    str += " dH:" + std::to_string(tiling_.get_dH());
    str += " dW:" + std::to_string(tiling_.get_dW());
    str += " divisorOverride:" + std::to_string(tiling_.get_divisorOverride());
    str += " countIncludePad:" + std::to_string(tiling_.get_countIncludePad());
    str += " channel:" + std::to_string(tiling_.get_channel());
    str += " blockFactor:" + std::to_string(tiling_.get_blockFactor());
    str += " blockTail:" + std::to_string(tiling_.get_blockTail());
    str += " totalIdx:" + std::to_string(tiling_.get_totalIdx());
    str += " coreNums:" + std::to_string(tiling_.get_coreNums());
    str += " inUbSize:" + std::to_string(tiling_.get_inUbSize());
    str += " outUbSize:" + std::to_string(tiling_.get_outUbSize());
    str += " isSigOut:" + std::to_string(tiling_.get_isSigOut());
    str += " tilingMode:" + std::to_string(tiling_.get_tilingMode());
    str += " onceOutNum:" + std::to_string(tiling_.get_onceOutNum());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// MaxPool3DBigKernelNDHWCTiling /////////////////////////////////
ge::graphStatus MaxPool3DBigKernelNDHWCTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize_, totalCoreNum_);
}

ge::graphStatus MaxPool3DBigKernelNDHWCTiling::GetShapeAttrsInfo()
{
    isMaxPool3D_ = true;
    return GetMaxPool3DShapeAttrsInfo(context_, inputData_);
}

//////////////////////////////// AvgPool3DBigKernelNDHWCTiling /////////////////////////////////
ge::graphStatus AvgPool3DBigKernelNDHWCTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize_, totalCoreNum_);
}

ge::graphStatus AvgPool3DBigKernelNDHWCTiling::GetShapeAttrsInfo()
{
    isMaxPool3D_ = false;
    return GetAvgPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DBigKernelNDHWCTiling, 13);
REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DBigKernelNDHWCTiling, 13);
}  // namespace optiling
