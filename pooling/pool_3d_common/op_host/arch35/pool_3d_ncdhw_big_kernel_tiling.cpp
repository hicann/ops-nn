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
 * \file pool_3d_ncdhw_big_kernel_tiling.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "pool_tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "pool_3d_ncdhw_big_kernel_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{

static constexpr uint64_t POOL_3D_TILING_KEY_BIG_KERNEL_NCHW = 511110;
static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t NUM256 = 256;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t TREE = 3;
static constexpr int64_t TWO = 2;
static constexpr int64_t ONE = 1;
bool Pool3DNcdhwBigKernelTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }
    if (inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] 
        < static_cast<int64_t>(coreNum_ * TWO)) {
        return true;
    }
    if (inputData_.kernelSize[D_DIM] * inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * inputData_.dtypeSize <
        NUM256) {
        return false;
    }
    return true;
}

void Pool3DNcdhwBigKernelTiling::DoUBTiling()
{
    totalIdx_ = inputData_.batches * inputData_.outShape[D_DIM] * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM];
    // coreNum已在tiling_base中校验过非0
    blockFactor_ = totalIdx_ / static_cast<int64_t>(coreNum_);
    blockTail_ = totalIdx_ % static_cast<int64_t>(coreNum_);
    if (blockFactor_ == 0) {
        coreNums_ = totalIdx_;
    } else {
        coreNums_ = static_cast<int64_t>(coreNum_);
    }
    isSigOut_ =
        (inputData_.outShape[D_DIM] == 1 && inputData_.outShape[H_DIM] == 1 && inputData_.outShape[W_DIM] == 1) ? 1 : 0;

    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(inputData_.dtypeSize) * OUT_BUFFER_LEN;
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    maxCount_ = ubAvailable / BUFFER_NUM - NUM256;
    int64_t divisor = ONE;
    if (inputData_.dtypeSize == TWO) {
        divisor = TREE;
    }
    maxCount_ = Ops::Base::FloorAlign(maxCount_ / divisor, ubBlockSize) / inputData_.dtypeSize;
}

ge::graphStatus Pool3DNcdhwBigKernelTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DNcdhwBigKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t Pool3DNcdhwBigKernelTiling::GetTilingKey() const
{
    return POOL_3D_TILING_KEY_BIG_KERNEL_NCHW;
}

ge::graphStatus Pool3DNcdhwBigKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Pool3DNcdhwBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNums_);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void Pool3DNcdhwBigKernelTiling::SetTilingData()
{
    tiling.set_dInDim(inputData_.inputShape[D_DIM]);
    tiling.set_hInDim(inputData_.inputShape[H_DIM]);
    tiling.set_wInDim(inputData_.inputShape[W_DIM]);
    tiling.set_dOutDim(inputData_.outShape[D_DIM]);
    tiling.set_hOutDim(inputData_.outShape[H_DIM]);
    tiling.set_wOutDim(inputData_.outShape[W_DIM]);
    tiling.set_kD(inputData_.kernelSize[D_DIM]);
    tiling.set_kH(inputData_.kernelSize[H_DIM]);
    tiling.set_kW(inputData_.kernelSize[W_DIM]);
    tiling.set_sD(inputData_.stride[D_DIM]);
    tiling.set_sH(inputData_.stride[H_DIM]);
    tiling.set_sW(inputData_.stride[W_DIM]);
    tiling.set_dD(inputData_.dilation[D_DIM]);
    tiling.set_dH(inputData_.dilation[H_DIM]);
    tiling.set_dW(inputData_.dilation[W_DIM]);
    tiling.set_fPad(inputData_.pad[FRONT_PAD_INDEX]);
    tiling.set_bkPad(inputData_.pad[BACKEND_PAD_INDEX]);
    tiling.set_tPad(inputData_.pad[TOP_PAD_INDEX]);
    tiling.set_bPad(inputData_.pad[BOTTOM_PAD_INDEX]);
    tiling.set_lPad(inputData_.pad[LEFT_PAD_INDEX]);
    tiling.set_rPad(inputData_.pad[RIGHT_PAD_INDEX]);
    tiling.set_divisorOverride(inputData_.divisorOverride);
    tiling.set_countIncludePad(int64_t(inputData_.countIncludePad));
    tiling.set_blockFactor(blockFactor_);
    tiling.set_blockTail(blockTail_);
    tiling.set_totalIdx(totalIdx_);
    tiling.set_coreNums(coreNums_);
    tiling.set_maxCount(maxCount_);
    tiling.set_isSigOut(isSigOut_);
}

void Pool3DNcdhwBigKernelTiling::DumpTilingInfo()
{
    std::string str;
    str += " dInDim:" + std::to_string(tiling.get_dInDim());
    str += " hInDim:" + std::to_string(tiling.get_hInDim());
    str += " wInDim:" + std::to_string(tiling.get_wInDim());
    str += " dOutDim:" + std::to_string(tiling.get_dOutDim());
    str += " hOutDim:" + std::to_string(tiling.get_hOutDim());
    str += " wOutDim:" + std::to_string(tiling.get_wOutDim());
    str += " kD:" + std::to_string(tiling.get_kD());
    str += " kH:" + std::to_string(tiling.get_kH());
    str += " kW:" + std::to_string(tiling.get_kW());
    str += " sD:" + std::to_string(tiling.get_sD());
    str += " sH:" + std::to_string(tiling.get_sH());
    str += " sW:" + std::to_string(tiling.get_sW());
    str += " dD:" + std::to_string(tiling.get_dD());
    str += " dH:" + std::to_string(tiling.get_dH());
    str += " dW:" + std::to_string(tiling.get_dW());
    str += " fPad:" + std::to_string(tiling.get_fPad());
    str += " bkPad:" + std::to_string(tiling.get_bkPad());
    str += " tPad:" + std::to_string(tiling.get_tPad());
    str += " bPad:" + std::to_string(tiling.get_bPad());
    str += " lPad:" + std::to_string(tiling.get_lPad());
    str += " rPad:" + std::to_string(tiling.get_rPad());
    str += " divisorOverride:" + std::to_string(tiling.get_divisorOverride());
    str += " countIncludePad:" + std::to_string(tiling.get_countIncludePad());
    str += " blockFactor:" + std::to_string(tiling.get_blockFactor());
    str += " blockTail:" + std::to_string(tiling.get_blockTail());
    str += " totalIdx:" + std::to_string(tiling.get_totalIdx());
    str += " coreNums:" + std::to_string(tiling.get_coreNums());
    str += " maxCount:" + std::to_string(tiling.get_maxCount());
    str += " isSigOut:" + std::to_string(tiling.get_isSigOut());
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// AvgPool3DNcdhwBigKernelTiling /////////////////////////////////
ge::graphStatus AvgPool3DNcdhwBigKernelTiling::GetPlatformInfo()
{
    return GetAvgPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPool3DNcdhwBigKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_POOL_TILING_TEMPLATE("AvgPool3D", AvgPool3DNcdhwBigKernelTiling, 11);

//////////////////////////////// MaxPool3DNcdhwBigKernelTiling /////////////////////////////////
ge::graphStatus MaxPool3DNcdhwBigKernelTiling::GetPlatformInfo()
{
    return GetMaxPool3DPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus MaxPool3DNcdhwBigKernelTiling::GetShapeAttrsInfo()
{
    return GetMaxPool3DShapeAttrsInfo(context_, inputData_);
}

REGISTER_OPS_POOL_TILING_TEMPLATE(MaxPool3D, MaxPool3DNcdhwBigKernelTiling, 11);
}  // namespace optiling