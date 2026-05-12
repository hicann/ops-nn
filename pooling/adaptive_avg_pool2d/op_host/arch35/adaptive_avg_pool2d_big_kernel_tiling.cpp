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
 * \file adaptive_avg_pool2d_tiling_big_kernel.cpp
 * \brief
 */

#include "adaptive_avg_pool2d_big_kernel_tiling.h"

namespace optiling {

static constexpr int64_t ADAPTIVE_AVG_POOL2D_BIG_KERNEL_THERSHOLD = 256;
static constexpr int64_t UB_MAX_INDICES_USE_COUNT = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_BYTES = 2;
static constexpr int64_t FLOAT32_BYTES = 4;
static constexpr int64_t STORE_ADD_SIZE = 1024;
static constexpr int64_t AVG_POOL2D_KERNEL_MININ_MAXOUT = 1024;
static constexpr int64_t AVG_POOL2D_ALIGN_BYTES = 32;
static constexpr int64_t AVG_POOL2D_SIMT_THREAD = 512;
static constexpr int64_t AVG_POOL2D_SIMT_USE_CORE = 16;
static constexpr int64_t AVG_POOL2D_BIGKERNEL_COPY_MODE_THERSHOLD = 8;

ge::graphStatus AdaptiveAvgPool2dBigKernelTiling::CheckOutputDtypeInfo()
{
    auto opNodeName = context_->GetNodeName();
    OP_LOGD(opNodeName, "CheckOutputDtypeInfo begin.");

    auto outputShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF((outputDtype != ge::DT_FLOAT && outputDtype != ge::DT_FLOAT16 && outputDtype != ge::DT_BF16),
        OP_LOGE(opNodeName, "output datatype only supports float, float16, bfloat16"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool2dBigKernelTiling::IsCapable()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dBigKernelTiling IsCapable check.");
    kernelHMin_ = CalKernelSizeOneDimMin(input_.hIn, input_.hOut);
    kernelWMin_ = CalKernelSizeOneDimMin(input_.wIn, input_.wOut);
    avgBigKernelInfo.kernelMinHW = kernelHMin_ * kernelWMin_;
    if (avgBigKernelInfo.kernelMinHW < ADAPTIVE_AVG_POOL2D_BIG_KERNEL_THERSHOLD) {
        return false;
    }
    if (avgBigKernelInfo.kernelMinHW >= AVG_POOL2D_KERNEL_MININ_MAXOUT) {
        return true;
    }
    if ((input_.hIn < input_.hOut || input_.wIn < input_.wOut)
        && ((input_.nIn * input_.cIn * input_.hOut * input_.wOut / AVG_POOL2D_SIMT_THREAD)
            > AVG_POOL2D_SIMT_USE_CORE)) {
        return false;
    }
    auto xDtypeSize = input_.xDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    if ((input_.hIn >= input_.hOut && input_.wIn >= input_.wOut)
        && (static_cast<int64_t>(kernelWMin_) < AVG_POOL2D_ALIGN_BYTES / static_cast<int64_t>(xDtypeSize))) {
        return false;
    }
    return true;
}

uint64_t AdaptiveAvgPool2dBigKernelTiling::GetTilingKey() const
{
    auto xDtypeSize = input_.xDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    uint64_t bigkernelCopyMode = TPL_BIG_KERNEL_COPYPAD;
    auto xAlignCount = AVG_POOL2D_ALIGN_BYTES / static_cast<int64_t>(xDtypeSize);
    if (static_cast<int64_t>(kernelWMin_) <= xAlignCount / AVG_POOL2D_BIGKERNEL_COPY_MODE_THERSHOLD) {
        bigkernelCopyMode = TPL_BIG_KERNEL_NDDMA;
    }
    return GET_TPL_TILING_KEY(TPL_BIG_KERNEL, TPL_INT32_UINT32, TPL_NC_FACTOR_64, bigkernelCopyMode);
}

void AdaptiveAvgPool2dBigKernelTiling::DoBlockTiling()
{
    avgBigKernelInfo.totalIdx = input_.nIn * input_.cIn * input_.hOut * input_.wOut;
    avgBigKernelInfo.blockFactor = avgBigKernelInfo.totalIdx / input_.coreNum;
    avgBigKernelInfo.blockTail = avgBigKernelInfo.totalIdx % input_.coreNum;

    if (avgBigKernelInfo.blockFactor == 0) {
        avgBigKernelInfo.coreNums = avgBigKernelInfo.totalIdx;
    } else {
        avgBigKernelInfo.coreNums = input_.coreNum;
    }

    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    auto xDtypeSize = input_.xDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    int64_t ubAvailable = input_.ubSize - xDtypeSize * UB_MAX_INDICES_USE_COUNT - FLOAT32_BYTES * STORE_ADD_SIZE;
    int64_t defaultMaxSize = Ops::Base::FloorAlign(ubAvailable / BUFFER_NUM, vRegSize);
    avgBigKernelInfo.maxCount = defaultMaxSize / FLOAT32_BYTES;
    avgBigKernelInfo.batchCount = avgBigKernelInfo.maxCount / avgBigKernelInfo.kernelMinHW;
}

void AdaptiveAvgPool2dBigKernelTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", coreNums: " << avgBigKernelInfo.coreNums;
    info << ", blockFactor: " << avgBigKernelInfo.blockFactor;
    info << ", blockTail: " << avgBigKernelInfo.blockTail;
    info << ", totalIdx: " << avgBigKernelInfo.totalIdx;
    info << ", maxCount: " << avgBigKernelInfo.maxCount;
    info << ", batchCount: " << avgBigKernelInfo.batchCount;
    info << std::endl;

    OP_LOGI("AdaptiveAvgPool2dBigKernel", "%s", info.str().c_str());
}

void AdaptiveAvgPool2dBigKernelTiling::SetTilingData()
{
    AdaptivePool2dBigKernelTilingData* tilingData = context_->GetTilingData<AdaptivePool2dBigKernelTilingData>();
    tilingData->nc = input_.nIn * input_.cIn;
    tilingData->hInDim = input_.hIn;
    tilingData->wInDim = input_.wIn;
    tilingData->hOutDim = input_.hOut;
    tilingData->wOutDim = input_.wOut;
    tilingData->blockFactor = avgBigKernelInfo.blockFactor;
    tilingData->blockTail = avgBigKernelInfo.blockTail;
    tilingData->totalIdx = avgBigKernelInfo.totalIdx;
    tilingData->coreNums = avgBigKernelInfo.coreNums;
    tilingData->maxCount = avgBigKernelInfo.maxCount;
    tilingData->batchCount = avgBigKernelInfo.batchCount;
}

ge::graphStatus AdaptiveAvgPool2dBigKernelTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dBigKernelTiling DoOpTiling start.");
    OP_CHECK_IF(CheckOutputDtypeInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool2d indices dtype unexpected"), return ge::GRAPH_FAILED);
    DoBlockTiling();
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool2dBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(avgBigKernelInfo.coreNums);
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveAvgPool2dBigKernelTiling::CalKernelSizeOneDimMin(uint64_t inSize, uint64_t outSize)
{
    uint64_t kernelSize;
    outSize = outSize == 0 ? 1 : outSize; // outSize = 0 is invalid, fallback to 1
    if (outSize > KERNEL_CALC_COUNT_THERSHOLD) {
        return (inSize + outSize - 1) / outSize + 1;
    }
    for (uint64_t i = 0; i < outSize; i++) {
        auto kernelLeft = (i * inSize) / outSize;
        auto kernelRight = ((i + 1) * inSize + outSize - 1) / outSize;
        auto kernelCurrent = kernelRight - kernelLeft;
        if (i == 0) {
            kernelSize = kernelCurrent;
        } else {
            kernelSize = kernelCurrent < kernelSize ? kernelCurrent : kernelSize;
        }
    }
    return kernelSize;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dBigKernelTiling, 1);

}