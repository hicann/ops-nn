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
 * \file adaptive_max_pool3d_tiling_big_kernel.cpp
 * \brief
 */

 #include "adaptive_max_pool3d_big_kernel_tiling.h"

namespace optiling {

static constexpr int64_t ADAPTIVE_MAX_POOL3D_BIG_KERNEL_THERSHOLD = 128;
static constexpr int64_t BIG_KERNEL_B2_MAX_COUNT = 32640;                 // FloorAlign(max_int16, VRegSize/sizeof(int16_t) int16最大值对)向下对齐
static constexpr int64_t UB_MAX_INDICES_USE_COUNT = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_BYTPES = 2;
static constexpr int64_t FLOAT32_BYTPES = 4;

bool AdaptiveMaxPool3dBigKernelTiling::IsCapable()
{
    // 按照搬运对齐的大小全载UB 和 kernelW<=16, 判断是否走当前模板
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigKernelTiling IsCapable check.");
    uint64_t kernelDMax = CalKernelSizeOneDimMax(input_.dIn, input_.dOut);
    uint64_t kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    uint64_t kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
    bigKernelInfo.kernelMaxDHW = kernelDMax * kernelHMax * kernelWMax;
    bool isCapable = bigKernelInfo.kernelMaxDHW >= ADAPTIVE_MAX_POOL3D_BIG_KERNEL_THERSHOLD;
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigKernelTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

uint64_t AdaptiveMaxPool3dBigKernelTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(TPL_MODE_1, TPL_DYTPE_0, TPL_MULTI_MODE_0, TPL_DATA_FORMAT_MODE_0);
}

void AdaptiveMaxPool3dBigKernelTiling::DoBlockTiling()
{
    bigKernelInfo.totalIdx = input_.nIn * input_.cIn * input_.dOut * input_.hOut * input_.wOut;
    bigKernelInfo.blockFactor = bigKernelInfo.totalIdx / input_.coreNum;
    bigKernelInfo.blockTail = bigKernelInfo.totalIdx % input_.coreNum;

    if (bigKernelInfo.blockFactor == 0) {
        bigKernelInfo.coreNums = bigKernelInfo.totalIdx;
    } else {
        bigKernelInfo.coreNums = input_.coreNum;
    }

    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    auto idxDtypeSize = ge::GetSizeByDataType(input_.indicesDtype);
    auto xDtypeSize = input_.xDtype == ge::DT_FLOAT ? FLOAT32_BYTPES : FLOAT16_BYTPES;
    int64_t ubAvailable = input_.ubSize - (xDtypeSize + idxDtypeSize) * UB_MAX_INDICES_USE_COUNT;
    int64_t defaultMaxSize = Ops::Base::FloorAlign(ubAvailable / BUFFER_NUM, vRegSize);
    bigKernelInfo.maxCount = input_.xDtype == ge::DT_FLOAT16 ? std::min(defaultMaxSize / FLOAT16_BYTPES, BIG_KERNEL_B2_MAX_COUNT) : 
                                defaultMaxSize / FLOAT32_BYTPES;
    bigKernelInfo.batchCount = bigKernelInfo.maxCount / bigKernelInfo.kernelMaxDHW;
}

void AdaptiveMaxPool3dBigKernelTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", coreNums: " << bigKernelInfo.coreNums;
    info << ", blockFactor: " << bigKernelInfo.blockFactor;
    info << ", blockTail: " << bigKernelInfo.blockTail;
    info << ", totalIdx: " << bigKernelInfo.totalIdx;
    info << ", maxCount: " << bigKernelInfo.maxCount;
    info << ", batchCount: " << bigKernelInfo.batchCount;
    info << std::endl;

    OP_LOGI("AdaptiveMaxPool3dBigKernel", "%s", info.str().c_str());
}
void AdaptiveMaxPool3dBigKernelTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData* tilingData = context_->GetTilingData<AdaptivePool3dBigKernelTilingData>();
    tilingData->nc = input_.nIn * input_.cIn;
    tilingData->dInDim = input_.dIn;
    tilingData->hInDim = input_.hIn;
    tilingData->wInDim = input_.wIn;
    tilingData->dOutDim = input_.dOut;
    tilingData->hOutDim = input_.hOut;
    tilingData->wOutDim = input_.wOut;
    tilingData->coreNums = bigKernelInfo.coreNums;
    tilingData->totalIdx = bigKernelInfo.totalIdx;
    tilingData->blockFactor = bigKernelInfo.blockFactor;
    tilingData->blockTail = bigKernelInfo.blockTail;
    tilingData->maxCount = bigKernelInfo.maxCount;
    tilingData->batchCount = bigKernelInfo.batchCount;
}

ge::graphStatus AdaptiveMaxPool3dBigKernelTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigKernelTiling DoOpTiling start.");
    OP_CHECK_IF(GetAndCheckIndicesDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveMaxPool3d indices dtype unexpected"), return ge::GRAPH_FAILED);
    DoBlockTiling();
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveMaxPool3dBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(bigKernelInfo.coreNums);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveMaxPool3d, AdaptiveMaxPool3dBigKernelTiling, 1);

} // namespace optiling