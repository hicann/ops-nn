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
 * \file max_pool_v3_big_kernel_nhwc_tiling.cpp
 * \brief big kernel imply for max_pool_v3 nhwc format
 */

#include "max_pool_v3_big_kernel_nhwc_tiling.h"

namespace optiling {
static constexpr uint64_t MAX_POOL_V3_TILING_KEY_BIG_KERNEL_NHWC = 411110;

static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t NO_SPLIT_KERNEL = 0;
static constexpr int64_t SPLIT_KERNEL_H = 1;
static constexpr int64_t SPLIT_KERNEL_W = 2;
static constexpr int64_t SPLIT_C = 3;
static constexpr int64_t MIN_OUT_BUFFER_LEN = 8192;

bool MaxPoolV3NHWCBigKernelTiling::IsCapable()
{
    if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        return true;
    }
    return false;
}

uint64_t MaxPoolV3NHWCBigKernelTiling::GetTilingKey() const
{
    return MAX_POOL_V3_TILING_KEY_BIG_KERNEL_NHWC;
}

void MaxPoolV3NHWCBigKernelTiling::DoUBTiling()
{
    totalIdx_ = inputData.batches * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    // coreNum已在tiling_base中校验过非0
    blockFactor_ = totalIdx_ / coreNum;
    blockTail_ = totalIdx_ % coreNum;
    if (blockFactor_ == 0) {
        coreNums_ = totalIdx_;
    } else {
        coreNums_ = coreNum;
    }
    isSigOut_ = (inputData.outShape[H_DIM] == 1 && inputData.outShape[W_DIM] == 1) ? 1 : 0;

    int64_t vRegSize = Ops::Base::GetVRegSize(context_) / dtypeSize;
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_) / dtypeSize;
    int64_t ubAvailable = static_cast<int64_t>(ubSize) / dtypeSize / BUFFER_NUM - vRegSize;
    int64_t oneOutChannel = std::min(MIN_OUT_BUFFER_LEN / dtypeSize, ubAvailable / DIGIT_TWO);
    int64_t channelAlign = Ops::Base::CeilAlign(inputData.channels, blockSize);
    int64_t oneOutChannelAlign = Ops::Base::CeilAlign(oneOutChannel, vRegSize);
    if (channelAlign > oneOutChannelAlign) {
        oneOutChannelAlign = Ops::Base::CeilAlign(channelAlign, vRegSize);
    }
    if ((inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM] * channelAlign + oneOutChannelAlign) <=
        ubAvailable) {
        inUbSize_ = inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM] * channelAlign;
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = NO_SPLIT_KERNEL;
    } else if ((inputData.kernelSize[W_DIM] * channelAlign + oneOutChannelAlign) <= ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_H;
    } else if ((channelAlign + oneOutChannelAlign) < ubAvailable) {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable - oneOutChannelAlign, vRegSize);
        outUbSize_ = oneOutChannelAlign;
        tilingMode_ = SPLIT_KERNEL_W;
    } else {
        inUbSize_ = Ops::Base::FloorAlign(ubAvailable / DIGIT_TWO, vRegSize);
        outUbSize_ = Ops::Base::FloorAlign(ubAvailable / DIGIT_TWO, vRegSize);
        tilingMode_ = SPLIT_C;
    }
    onceOutNum_ = Ops::Base::FloorDiv(outUbSize_, channelAlign);
    outUbSize_ = outUbSize_ + vRegSize;
}

void MaxPoolV3NHWCBigKernelTiling::SetTilingData()
{
    tiling.set_hInDim(inputData.inputShape[H_DIM]);
    tiling.set_wInDim(inputData.inputShape[W_DIM]);
    tiling.set_hOutDim(inputData.outShape[H_DIM]);
    tiling.set_wOutDim(inputData.outShape[W_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_tPad(inputData.pad[TOP_PAD_INDEX]);
    tiling.set_bPad(inputData.pad[BOTTOM_PAD_INDEX]);
    tiling.set_lPad(inputData.pad[LEFT_PAD_INDEX]);
    tiling.set_rPad(inputData.pad[RIGHT_PAD_INDEX]);
    tiling.set_channel(inputData.channels);
    tiling.set_blockFactor(blockFactor_);
    tiling.set_blockTail(blockTail_);
    tiling.set_totalIdx(totalIdx_);
    tiling.set_coreNums(coreNums_);
    tiling.set_inUbSize(inUbSize_);
    tiling.set_outUbSize(outUbSize_);
    tiling.set_isSigOut(isSigOut_);
    tiling.set_tilingMode(tilingMode_);
    tiling.set_onceOutNum(onceOutNum_);
}

ge::graphStatus MaxPoolV3NHWCBigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3NHWCBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MaxPoolV3NHWCBigKernelTiling::DumpTilingInfo()
{
    std::string str;
    str += " hInDim:" + std::to_string(tiling.get_hInDim());
    str += " wInDim:" + std::to_string(tiling.get_wInDim());
    str += " hOutDim:" + std::to_string(tiling.get_hOutDim());
    str += " wOutDim:" + std::to_string(tiling.get_wOutDim());
    str += " kH:" + std::to_string(tiling.get_kH());
    str += " kW:" + std::to_string(tiling.get_kW());
    str += " sH:" + std::to_string(tiling.get_sH());
    str += " sW:" + std::to_string(tiling.get_sW());
    str += " tPad:" + std::to_string(tiling.get_tPad());
    str += " bPad:" + std::to_string(tiling.get_bPad());
    str += " lPad:" + std::to_string(tiling.get_lPad());
    str += " rPad:" + std::to_string(tiling.get_rPad());
    str += " channel:" + std::to_string(tiling.get_channel());
    str += " blockFactor:" + std::to_string(tiling.get_blockFactor());
    str += " blockTail:" + std::to_string(tiling.get_blockTail());
    str += " totalIdx:" + std::to_string(tiling.get_totalIdx());
    str += " coreNums:" + std::to_string(tiling.get_coreNums());
    str += " inUbSize:" + std::to_string(tiling.get_inUbSize());
    str += " outUbSize:" + std::to_string(tiling.get_outUbSize());
    str += " isSigOut:" + std::to_string(tiling.get_isSigOut());
    str += " tilingMode:" + std::to_string(tiling.get_tilingMode());
    str += " onceOutNum:" + std::to_string(tiling.get_onceOutNum());
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolV3, MaxPoolV3NHWCBigKernelTiling, 4);
REGISTER_OPS_TILING_TEMPLATE(MaxPoolV2, MaxPoolV2NHWCBigKernelTiling, 4);

} // namespace optiling
