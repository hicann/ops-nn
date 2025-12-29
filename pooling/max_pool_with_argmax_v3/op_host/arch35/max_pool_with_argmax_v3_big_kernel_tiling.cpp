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
 * \file max_pool_with_argmax_v3_big_kernel_tiling.cpp
 * \brief big kernel imply for max_pool_with_argmax
 */

#include "op_util.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool_with_argmax_v3_big_kernel_tiling.h"

static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_NCHW = 311110;

static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t MIN_COUNT = 1024;
static constexpr int64_t BYTES_FOUR = 4;
static constexpr int64_t BYTES_EIGHT = 8;
static constexpr int64_t KW_THRESHOLD = 128;
using namespace AscendC;

namespace optiling {

bool MaxPoolWithArgmaxV3BigKernelTiling::IsCapable()
{
    int64_t ubAvailable = ubSize - (BYTES_FOUR + BYTES_EIGHT) * OUT_BUFFER_LEN;
    maxCount_ = ubAvailable / BUFFER_NUM;
    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    maxCount_ = Ops::Base::FloorAlign(maxCount_, vRegSize);
    int64_t dtypeSize = ge::GetSizeByDataType(dtype);
    OP_CHECK_IF(
        dtypeSize <= 0, OP_LOGE(context_, "dtypeSize must be greater than 0, dtypeSize: %ld", dtypeSize), return false);
    maxCount_ = maxCount_ / dtypeSize;
    if (inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1 && maxCount_ > MIN_COUNT &&
        inputData.inputFormat == ge::Format::FORMAT_NCHW && inputData.kernelSize[W_DIM] * dtypeSize > KW_THRESHOLD) {
        return true;
    }
    return false;
}

uint64_t MaxPoolWithArgmaxV3BigKernelTiling::GetTilingKey() const
{
    return MAX_POOL_WITH_ARGMAX_V3_TILING_KEY_BIG_KERNEL_NCHW;
}

void MaxPoolWithArgmaxV3BigKernelTiling::DoUBTiling()
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
}

void MaxPoolWithArgmaxV3BigKernelTiling::SetTilingData()
{
    tiling.set_hInDim(inputData.inputShape[H_DIM]);
    tiling.set_wInDim(inputData.inputShape[W_DIM]);
    tiling.set_hOutDim(inputData.outShape[H_DIM]);
    tiling.set_wOutDim(inputData.outShape[W_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_pH(inputData.pad[H_DIM]);
    tiling.set_pW(inputData.pad[W_DIM]);
    tiling.set_dH(inputData.dilation[H_DIM]);
    tiling.set_dW(inputData.dilation[W_DIM]);
    tiling.set_blockFactor(blockFactor_);
    tiling.set_blockTail(blockTail_);
    tiling.set_totalIdx(totalIdx_);
    tiling.set_coreNums(coreNums_);
    tiling.set_maxCount(maxCount_);
    tiling.set_isSigOut(isSigOut_);
}

ge::graphStatus MaxPoolWithArgmaxV3BigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3BigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MaxPoolWithArgmaxV3BigKernelTiling::DumpTilingInfo()
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
    str += " pH:" + std::to_string(tiling.get_pH());
    str += " pW:" + std::to_string(tiling.get_pW());
    str += " blockFactor:" + std::to_string(tiling.get_blockFactor());
    str += " blockTail:" + std::to_string(tiling.get_blockTail());
    str += " totalIdx:" + std::to_string(tiling.get_totalIdx());
    str += " coreNums:" + std::to_string(tiling.get_coreNums());
    str += " maxCount:" + std::to_string(tiling.get_maxCount());
    str += " isSigOut:" + std::to_string(tiling.get_isSigOut());
    OP_LOGI(context_, "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolWithArgmaxV3, MaxPoolWithArgmaxV3BigKernelTiling, 6);

} // namespace optiling
