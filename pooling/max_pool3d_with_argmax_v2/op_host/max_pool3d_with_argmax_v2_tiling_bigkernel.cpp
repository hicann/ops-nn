/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_with_argmax_v2_tiling_bigkernel.cpp
 * \brief
 */

#include "max_pool3d_with_argmax_v2_tiling_base.h"

// 3, splitD=1, splitH=1, splitW=1, splitKernel = 1, dtype=float
constexpr uint64_t TILING_KEY_BIG_KERNEL_FLOAT = 311110;
// 3, splitD=1, splitH=1, splitW=0, splitKernel = 1, dtype=half
constexpr uint64_t TILING_KEY_BIG_KERNEL_HALF = 311111;
// 3, splitD=1, splitH=1, splitW=0, splitKernel = 1, dtype=bfloat16
constexpr uint64_t TILING_KEY_BIG_KERNEL_BF16 = 311112;


namespace optiling {

bool MaxPool3DWithArgmaxV2BigKernelTiling::IsCapable()
{
    if (inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1) {
        return true;
    }
    return false;
}

uint64_t MaxPool3DWithArgmaxV2BigKernelTiling::GetTilingKey() const
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_BIG_KERNEL_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_BIG_KERNEL_HALF;
    } else {
        return TILING_KEY_BIG_KERNEL_BF16;
    }
}

void MaxPool3DWithArgmaxV2BigKernelTiling::DoUBTiling()
{
    totalIdx = inputData.batches * inputData.outShape[D_DIM] * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    blockFactor = totalIdx / coreNum;
    blockTail = totalIdx % coreNum;
    if (blockFactor == 0UL) {
        coreNums = totalIdx;
    } else {
        coreNums = coreNum;
    }
}

void MaxPool3DWithArgmaxV2BigKernelTiling::SetTilingData()
{
    array<uint32_t, DHW_DIMS> tmpInputShape{
        static_cast<uint32_t>(inputData.inputShape[D_DIM]), static_cast<uint32_t>(inputData.inputShape[H_DIM]),
        static_cast<uint32_t>(inputData.inputShape[W_DIM])};
    array<uint32_t, DHW_DIMS> tmpOutShape{
        static_cast<uint32_t>(inputData.outShape[D_DIM]), static_cast<uint32_t>(inputData.outShape[H_DIM]), static_cast<uint32_t>(inputData.outShape[W_DIM])};
    tiling.set_inputShapes(&(tmpInputShape[0]));
    tiling.set_outShapes(&(tmpOutShape[0]));
    tiling.set_kD((uint32_t)inputData.kernelSize[D_DIM]);
    tiling.set_kH((uint32_t)inputData.kernelSize[H_DIM]);
    tiling.set_kW((uint32_t)inputData.kernelSize[W_DIM]);
    tiling.set_sD((uint32_t)inputData.stride[D_DIM]);
    tiling.set_sH((uint32_t)inputData.stride[H_DIM]);
    tiling.set_sW((uint32_t)inputData.stride[W_DIM]);
    tiling.set_pD((uint32_t)inputData.pad[D_DIM]);
    tiling.set_pH((uint32_t)inputData.pad[H_DIM]);
    tiling.set_pW((uint32_t)inputData.pad[W_DIM]);
    tiling.set_dD((uint32_t)inputData.dilation[D_DIM]);
    tiling.set_dH((uint32_t)inputData.dilation[H_DIM]);
    tiling.set_dW((uint32_t)inputData.dilation[W_DIM]);
    tiling.set_blockFactor(blockFactor);
    tiling.set_blockTail(blockTail);
    tiling.set_totalIdx(totalIdx);
    tiling.set_coreNums(coreNums);
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2BigKernelTiling, 1);

} // namespace optiling
