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

static constexpr int64_t ADAPTIVE_AVG_POOL2D_BIG_KERNEL_THERSHOLD = 128;
static constexpr int64_t UB_MAX_INDICES_USE_COUNT = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_BYTES = 2;
static constexpr int64_t FLOAT32_BYTES = 4;
static constexpr int64_t STORE_ADD_SIZE = 1024;

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
    // 按照搬运对齐的大小全载UB 和 kernelW<=16, 判断是否走当前模板
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dBigKernelTiling IsCapable check.");
    uint64_t kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    uint64_t kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
    avgBigKernelInfo.kernelMaxHW = kernelHMax * kernelWMax;
    bool isCapable = avgBigKernelInfo.kernelMaxHW >= ADAPTIVE_AVG_POOL2D_BIG_KERNEL_THERSHOLD;
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool2dBigKernelTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

uint64_t AdaptiveAvgPool2dBigKernelTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(TPL_BIG_KERNEL, TPL_INT32_UINT32);
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
    avgBigKernelInfo.batchCount = avgBigKernelInfo.maxCount / avgBigKernelInfo.kernelMaxHW;
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

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool2d, AdaptiveAvgPool2dBigKernelTiling, 1);

}