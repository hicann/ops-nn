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
 * \file max_pool_v3_big_kernel_tiling.cpp
 * \brief big kernel imply for max_pool_v3
 */

#include "max_pool_v3_big_kernel_tiling.h"

namespace optiling {
static constexpr uint64_t MAX_POOL_V3_TILING_KEY_BIG_KERNEL_NCHW = 311110;

static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;

bool MaxPoolV3BigKernelTiling::IsCapable()
{
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        return true;
    }
    return false;
}

uint64_t MaxPoolV3BigKernelTiling::GetTilingKey() const
{
    return MAX_POOL_V3_TILING_KEY_BIG_KERNEL_NCHW;
}

void MaxPoolV3BigKernelTiling::DoUBTiling()
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

    int64_t ubAvailable = static_cast<int64_t>(ubSize) - static_cast<int64_t>(dtypeSize) * OUT_BUFFER_LEN;
    maxCount_ = ubAvailable / BUFFER_NUM;
    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    maxCount_ = Ops::Base::FloorAlign(maxCount_, vRegSize);
    maxCount_ = maxCount_ / dtypeSize;
}

void MaxPoolV3BigKernelTiling::SetTilingData()
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
    tiling.set_blockFactor(blockFactor_);
    tiling.set_blockTail(blockTail_);
    tiling.set_totalIdx(totalIdx_);
    tiling.set_coreNums(coreNums_);
    tiling.set_maxCount(maxCount_);
    tiling.set_isSigOut(isSigOut_);
}

ge::graphStatus MaxPoolV3BigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MaxPoolV3BigKernelTiling::DumpTilingInfo()
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
    str += " blockFactor:" + std::to_string(tiling.get_blockFactor());
    str += " blockTail:" + std::to_string(tiling.get_blockTail());
    str += " totalIdx:" + std::to_string(tiling.get_totalIdx());
    str += " coreNums:" + std::to_string(tiling.get_coreNums());
    str += " maxCount:" + std::to_string(tiling.get_maxCount());
    str += " isSigOut:" + std::to_string(tiling.get_isSigOut());
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolV3, MaxPoolV3BigKernelTiling, 2);
REGISTER_OPS_TILING_TEMPLATE(MaxPoolV2, MaxPoolV2BigKernelTiling, 2);

} // namespace optiling
