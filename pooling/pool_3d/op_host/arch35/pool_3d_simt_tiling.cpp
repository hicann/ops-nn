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
 * \file pool_3d_simt_tiling.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "pool_tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "pool_3d_simt_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
static constexpr uint64_t POOL_3D_TILING_KEY_SIMT_NCDHW_INT32 = 911100;
static constexpr uint64_t POOL_3D_TILING_KEY_SIMT_NDHWC_INT32 = 911101;
static constexpr uint64_t POOL_3D_TILING_KEY_SIMT_NCDHW_INT64 = 911110;
static constexpr uint64_t POOL_3D_TILING_KEY_SIMT_NDHWC_INT64 = 911111;

bool Pool3DSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus Pool3DSimtTiling::DoOpTiling()
{
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t Pool3DSimtTiling::GetTilingKey() const
{
    int64_t xSize = inputData.batches * inputData.channels * inputData.inputShape[D_DIM] * inputData.inputShape[H_DIM] * inputData.inputShape[W_DIM];
    int64_t ySize = inputData.batches * inputData.channels * inputData.outShape[D_DIM] * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        if (xSize <= INT32_MAX && ySize <= INT32_MAX) {
            return POOL_3D_TILING_KEY_SIMT_NCDHW_INT32;
        }
        return POOL_3D_TILING_KEY_SIMT_NCDHW_INT64;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        if (xSize <= INT32_MAX) {
            return POOL_3D_TILING_KEY_SIMT_NDHWC_INT32;
        }
        return POOL_3D_TILING_KEY_SIMT_NDHWC_INT64;
    }
    return 0;
}

ge::graphStatus Pool3DSimtTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DSimtTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);
    ubSize = ubSize - DCACHE_SIZE;
    auto res = context_->SetLocalMemorySize(ubSize);
    OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize),
                    return ge::GRAPH_FAILED);
    tiling_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DSimtTiling::SetTilingData()
{
    tiling_.set_nDim(inputData.batches);
    tiling_.set_cDim(inputData.channels);
    tiling_.set_dInDim(inputData.inputShape[D_DIM]);
    tiling_.set_hInDim(inputData.inputShape[H_DIM]);
    tiling_.set_wInDim(inputData.inputShape[W_DIM]);
    tiling_.set_dOutDim(inputData.outShape[D_DIM]);
    tiling_.set_hOutDim(inputData.outShape[H_DIM]);
    tiling_.set_wOutDim(inputData.outShape[W_DIM]);
    tiling_.set_kD(inputData.kernelSize[D_DIM]);
    tiling_.set_kH(inputData.kernelSize[H_DIM]);
    tiling_.set_kW(inputData.kernelSize[W_DIM]);
    tiling_.set_sD(inputData.stride[D_DIM]);
    tiling_.set_sH(inputData.stride[H_DIM]);
    tiling_.set_sW(inputData.stride[W_DIM]);
    tiling_.set_dD(inputData.dilation[D_DIM]);
    tiling_.set_dH(inputData.dilation[H_DIM]);
    tiling_.set_dW(inputData.dilation[W_DIM]);
    tiling_.set_fPad(inputData.pad[FRONT_PAD_INDEX]);
    tiling_.set_bkPad(inputData.pad[BACKEND_PAD_INDEX]);
    tiling_.set_tPad(inputData.pad[TOP_PAD_INDEX]);
    tiling_.set_bPad(inputData.pad[BOTTOM_PAD_INDEX]);
    tiling_.set_lPad(inputData.pad[LEFT_PAD_INDEX]);
    tiling_.set_rPad(inputData.pad[RIGHT_PAD_INDEX]);
    tiling_.set_divisorOverride(inputData.divisorOverride);
    tiling_.set_countIncludePad(inputData.countIncludePad ? 1 : 0);
}

void Pool3DSimtTiling::DumpTilingInfo()
{
    std::string str;
    str += " nDim:" + std::to_string(tiling_.get_nDim());
    str += " cDim:" + std::to_string(tiling_.get_cDim());
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
    str += " dD:" + std::to_string(tiling_.get_dD());
    str += " dH:" + std::to_string(tiling_.get_dH());
    str += " dW:" + std::to_string(tiling_.get_dW());
    str += " fPad:" + std::to_string(tiling_.get_fPad());
    str += " bkPad:" + std::to_string(tiling_.get_bkPad());
    str += " tPad:" + std::to_string(tiling_.get_tPad());
    str += " bPad:" + std::to_string(tiling_.get_bPad());
    str += " lPad:" + std::to_string(tiling_.get_lPad());
    str += " rPad:" + std::to_string(tiling_.get_rPad());
    str += " divisorOverride:" + std::to_string(tiling_.get_divisorOverride());
    str += " countIncludePad:" + std::to_string(tiling_.get_countIncludePad());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// AvgPool3DSimtTiling /////////////////////////////////
ge::graphStatus AvgPool3DSimtTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus AvgPool3DSimtTiling::GetShapeAttrsInfo()
{
    return GetAvgPool3DShapeAttrsInfo(context_, inputData);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DSimtTiling, 19);

//////////////////////////////// MaxPool3DSimtTiling /////////////////////////////////
ge::graphStatus MaxPool3DSimtTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus MaxPool3DSimtTiling::GetShapeAttrsInfo()
{
    return GetMaxPool3DShapeAttrsInfo(context_, inputData);
}

REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DSimtTiling, 19);
}  // namespace optiling