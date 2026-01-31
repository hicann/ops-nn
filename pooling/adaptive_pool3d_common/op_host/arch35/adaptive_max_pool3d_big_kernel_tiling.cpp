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
static constexpr int64_t KERNEL_CALC_COUNT_THERSHOLD = 10000;
static constexpr int64_t BIG_KERNEL_TILING_KEY = 2000;
static constexpr int64_t UB_MAX_INDICES_USE_COUNT = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_BYTPES = 2;
static constexpr int64_t FLOAT32_BYTPES = 4;
static constexpr int64_t INT32_MAX_SUPPORT_VALUE = 2147483647UL;
static constexpr int64_t MULTI_NO_SPILT_COPY_FLAG = 1;

ge::graphStatus AdaptiveMaxPool3dBigKernelTiling::GetIndexDtypeInfo()
{
    auto opNodeName = context_->GetNodeName();
    OP_LOGD(opNodeName, "GetShapeAttrsInfo begin.");

    auto outputIndices = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputIndices);
    auto outputIndicesDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputIndicesDesc);
    auto indicesDtype = outputIndicesDesc->GetDataType();
    OP_CHECK_IF((indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64),
        OP_LOGE(opNodeName, "output indices datatype only support int32, int64"), return ge::GRAPH_FAILED);
    int64_t outputIndicesMaxValue = input_.dIn * input_.hIn * input_.wIn;
    OP_CHECK_IF((outputIndicesMaxValue > INT32_MAX_SUPPORT_VALUE && indicesDtype == ge::DT_INT32),
        OP_LOGE(opNodeName, "max index exceeds the max value of int32"), return ge::GRAPH_FAILED);
    bigKernelInfo.indicesDtype = indicesDtype;
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3dBigKernelTiling::IsCapable()
{
    // 按照搬运对齐的大小全载UB 和 kernelW<=16, 判断是否走当前模板
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigKernelTiling IsCapable check.");
    int64_t kernelDMax = CalKernelMax(input_.dIn, input_.dOut);
    int64_t kernelHMax = CalKernelMax(input_.hIn, input_.hOut);
    int64_t kernelWMax = CalKernelMax(input_.wIn, input_.wOut);
    bigKernelInfo.kernelMaxDHW = kernelDMax * kernelHMax * kernelWMax;
    bool isCapable = bigKernelInfo.kernelMaxDHW >= ADAPTIVE_MAX_POOL3D_BIG_KERNEL_THERSHOLD;
    OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigKernelTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

int64_t AdaptiveMaxPool3dBigKernelTiling::CalKernelMax(int64_t inSize, int64_t outSize)
{
    // 计算kernel的max值
    int64_t kernelSize = 1;
    outSize = outSize == 0 ? 1 : outSize;
    if (outSize > KERNEL_CALC_COUNT_THERSHOLD) {
        return (inSize + outSize - 1) / outSize + 1; // 防止计算时间过长
    }
    for (int64_t i = 0; i < outSize; i++) {
        auto kernelLeft = (i * inSize) / outSize;
        auto kernelRight = ((i + 1) * inSize + outSize - 1) / outSize;
        auto kernelCurrent = kernelRight - kernelLeft;
        kernelSize = kernelCurrent > kernelSize ? kernelCurrent : kernelSize;
    }
    return kernelSize;
}

uint64_t AdaptiveMaxPool3dBigKernelTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(TPL_MODE_1, TPL_DYTPE_0, TPL_MULTI_MODE_0);
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
    auto idxDtypeSize = ge::GetSizeByDataType(bigKernelInfo.indicesDtype);
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
    OP_CHECK_IF(GetIndexDtypeInfo() != ge::GRAPH_SUCCESS,
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