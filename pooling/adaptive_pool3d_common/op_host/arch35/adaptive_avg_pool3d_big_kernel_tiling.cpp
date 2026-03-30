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
 * \file adaptive_avg_pool3d_tiling_big_kernel.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_avg_pool3d_big_kernel_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

namespace optiling {

static constexpr int64_t ADAPTIVE_AVG_POOL3D_BIG_KERNEL_THERSHOLD = 128;
static constexpr int64_t UB_MAX_INDICES_USE_COUNT = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_BYTPES = 2;
static constexpr int64_t FLOAT32_BYTPES = 4;
static constexpr int64_t STORE_ADD_SIZE = 1024;

ge::graphStatus AdaptiveAvgPool3dBigKernelTiling::CheckOutputDtypeInfo()
{
    auto opNodeName = context_->GetNodeName();
    OP_LOGD(opNodeName, "CheckOutputDtypeInfo begin.");

    auto outputShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF((outputDtype != ge::DT_FLOAT && outputDtype != ge::DT_FLOAT16 && outputDtype != ge::DT_BF16),
        OP_LOGE(opNodeName, "output datatype only support only supports float, float16, bfloat16"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveAvgPool3dBigKernelTiling::IsCapable()
{
    OP_TILING_CHECK(
        GetAndCheckDataFormat() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetDataFormatAttrInfo fail."),
        return ge::GRAPH_FAILED);
    // 按照搬运对齐的大小全载UB, 判断是否走当前模板
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dBigKernelTiling IsCapable check.");
    uint64_t kernelDMax = CalKernelSizeOneDimMax(input_.dIn, input_.dOut);
    uint64_t kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
    uint64_t kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
    avgBigKernelInfo.kernelMaxDHW = kernelDMax * kernelHMax * kernelWMax;
    bool isCapable = (avgBigKernelInfo.kernelMaxDHW >= ADAPTIVE_AVG_POOL3D_BIG_KERNEL_THERSHOLD) && (input_.dataFormat == ge::Format::FORMAT_NCDHW);
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dBigKernelTiling IsCapable check: %s", isCapable ? "true" : "false");
    return isCapable;
}

uint64_t AdaptiveAvgPool3dBigKernelTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(TPL_MODE_1, TPL_DTYPE_0, TPL_MULTI_MODE_0, TPL_DATA_FORMAT_MODE_1);
}

void AdaptiveAvgPool3dBigKernelTiling::DoBlockTiling()
{
    avgBigKernelInfo.totalIdx = input_.nIn * input_.cIn * input_.dOut * input_.hOut * input_.wOut;
    avgBigKernelInfo.blockFactor = avgBigKernelInfo.totalIdx / input_.coreNum;
    avgBigKernelInfo.blockTail = avgBigKernelInfo.totalIdx % input_.coreNum;

    if (avgBigKernelInfo.blockFactor == 0) {
        avgBigKernelInfo.coreNums = avgBigKernelInfo.totalIdx;
    } else {
        avgBigKernelInfo.coreNums = input_.coreNum;
    }

    int64_t vRegSize = Ops::Base::GetVRegSize(context_);
    auto xDtypeSize = input_.xDtype == ge::DT_FLOAT ? FLOAT32_BYTPES : FLOAT16_BYTPES;
    int64_t ubAvailable = input_.ubSize - xDtypeSize * UB_MAX_INDICES_USE_COUNT - FLOAT32_BYTPES * STORE_ADD_SIZE; 
    int64_t defaultMaxSize = Ops::Base::FloorAlign(ubAvailable / BUFFER_NUM, vRegSize);
    avgBigKernelInfo.maxCount = defaultMaxSize / FLOAT32_BYTPES;
    avgBigKernelInfo.batchCount = avgBigKernelInfo.maxCount / avgBigKernelInfo.kernelMaxDHW;
}

void AdaptiveAvgPool3dBigKernelTiling::PrintTilingData() const
{
    std::ostringstream info;
    info << "nc: " << input_.nIn * input_.cIn;
    info << ", dInDim: " << input_.dIn;
    info << ", hInDim: " << input_.hIn;
    info << ", wInDim: " << input_.wIn;
    info << ", dOutDim: " << input_.dOut;
    info << ", hOutDim: " << input_.hOut;
    info << ", wOutDim: " << input_.wOut;
    info << ", coreNums: " << avgBigKernelInfo.coreNums;
    info << ", blockFactor: " << avgBigKernelInfo.blockFactor;
    info << ", blockTail: " << avgBigKernelInfo.blockTail;
    info << ", totalIdx: " << avgBigKernelInfo.totalIdx;
    info << ", maxCount: " << avgBigKernelInfo.maxCount;
    info << ", batchCount: " << avgBigKernelInfo.batchCount;
    info << std::endl;

    OP_LOGI("AdaptiveAvgPool3dBigKernel", "%s", info.str().c_str());
}
void AdaptiveAvgPool3dBigKernelTiling::SetTilingData()
{
    AdaptivePool3DTiling::AdaptivePool3dBigKernelTilingData* tilingData = context_->GetTilingData<AdaptivePool3dBigKernelTilingData>();
    tilingData->nc = input_.nIn * input_.cIn;
    tilingData->dInDim = input_.dIn;
    tilingData->hInDim = input_.hIn;
    tilingData->wInDim = input_.wIn;
    tilingData->dOutDim = input_.dOut;
    tilingData->hOutDim = input_.hOut;
    tilingData->wOutDim = input_.wOut;
    tilingData->blockFactor = avgBigKernelInfo.blockFactor;
    tilingData->blockTail = avgBigKernelInfo.blockTail;
    tilingData->totalIdx = avgBigKernelInfo.totalIdx;
    tilingData->coreNums = avgBigKernelInfo.coreNums;
    tilingData->maxCount = avgBigKernelInfo.maxCount;
    tilingData->batchCount = avgBigKernelInfo.batchCount;
}

ge::graphStatus AdaptiveAvgPool3dBigKernelTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "AdaptiveAvgPool3dBigKernelTiling DoOpTiling start.");
    OP_CHECK_IF(CheckOutputDtypeInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "AdaptiveAvgPool3d indices dtype unexpected"), return ge::GRAPH_FAILED);
    DoBlockTiling();
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveAvgPool3dBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(avgBigKernelInfo.coreNums);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AdaptiveAvgPool3d, AdaptiveAvgPool3dBigKernelTiling, 1);

}