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
* \file avg_pool_big_kernel_tiling.cpp
* \brief big kernel imply for avg_pool
*/

#include "platform_util.h"
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "avg_pool_big_kernel_tiling.h"

namespace optiling
{
using namespace AscendC;

static constexpr uint64_t AVG_POOL_BIG_KERNEL_FORMAT_NCHW = 511110;

static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t NUM256 = 256;
static constexpr int64_t THREE = 3;
static constexpr int64_t TWO = 2;
static constexpr int64_t ONE = 1;
static constexpr int64_t EIGHT = 8;
static constexpr int64_t SIMT_STRIDE_THREHOLD = 32;
static constexpr int64_t SIMT_kSIZE_THREHOLD = 5000;
static constexpr int64_t NCHW_THREHOLD = 8400;

bool AvgPoolCommonBigKernelTiling::IsCapable()
{
    if (inputData_.inputFormat != ge::Format::FORMAT_NCHW) {
        return false;
    }
    if ((inputData_.stride[W_DIM] < inputData_.kernelSize[W_DIM] / EIGHT || inputData_.stride[W_DIM] < SIMT_STRIDE_THREHOLD) &&
        (inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] < SIMT_kSIZE_THREHOLD) &&
        (inputData_.batches * inputData_.channels * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] > NCHW_THREHOLD)) {
        return false;
    }
    if (inputData_.batches * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM] < static_cast<int64_t>(coreNum_ * TWO)) {
        return true;
    }
    if (inputData_.kernelSize[H_DIM] * inputData_.kernelSize[W_DIM] * inputData_.dtypeSize < NUM256) {
        return false;
    }
    return true;
}

uint64_t AvgPoolCommonBigKernelTiling::GetTilingKey() const
{
    OP_LOGD(context_, "AvgPoolCommonBigKernelTiling GetTilingKey is %" PRIu64, AVG_POOL_BIG_KERNEL_FORMAT_NCHW);
    return AVG_POOL_BIG_KERNEL_FORMAT_NCHW;
}

void AvgPoolCommonBigKernelTiling::DoUBTiling()
{
    totalIdx_ = inputData_.batches * inputData_.outShape[H_DIM] * inputData_.outShape[W_DIM];
    // coreNum已在tiling_base中校验过非0
    blockFactor_ = totalIdx_ / static_cast<int64_t>(coreNum_);
    blockTail_ = totalIdx_ % static_cast<int64_t>(coreNum_);
    coreNum_ = blockFactor_ == 0 ? totalIdx_ : static_cast<int64_t>(coreNum_);
    isSigOut_ = (inputData_.outShape[H_DIM] == 1 && inputData_.outShape[W_DIM] == 1) ? 1 : 0;

    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(inputData_.dtypeSize) * OUT_BUFFER_LEN;
    int64_t ubBlockSize = platform::GetUbBlockSize(context_);
    maxCount_ = ubAvailable / BUFFER_NUM - NUM256;
    int64_t divisor = ONE;
    if (inputData_.dtypeSize == TWO) {
        divisor = THREE;
    }
    maxCount_ = Ops::Base::FloorAlign(maxCount_ / divisor, ubBlockSize) / inputData_.dtypeSize;
}

void AvgPoolCommonBigKernelTiling::SetTilingData()
{
    OP_LOGD(context_, "AvgPoolCommonBigKernelTiling SetTilingData begin.");
    AvgPool::AvgPoolBigKernelTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolBigKernelTilingData>();

    tilingData->hInDim = inputData_.inputShape[H_DIM];
    tilingData->wInDim = inputData_.inputShape[W_DIM];
    tilingData->hOutDim = inputData_.outShape[H_DIM];
    tilingData->wOutDim = inputData_.outShape[W_DIM];
    tilingData->kH = inputData_.kernelSize[H_DIM];
    tilingData->kW = inputData_.kernelSize[W_DIM];
    tilingData->sH = inputData_.stride[H_DIM];
    tilingData->sW = inputData_.stride[W_DIM];
    tilingData->tPad = inputData_.pad[TOP_PAD_INDEX];
    tilingData->bPad = inputData_.pad[BOTTOM_PAD_INDEX];
    tilingData->lPad = inputData_.pad[LEFT_PAD_INDEX];
    tilingData->rPad = inputData_.pad[RIGHT_PAD_INDEX];
    tilingData->divisorOverride = inputData_.divisorOverride;
    tilingData->countIncludePad = static_cast<int64_t>(inputData_.countIncludePad);
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->totalIdx = totalIdx_;
    tilingData->coreNums = coreNum_;
    tilingData->maxCount = maxCount_;
    tilingData->isSigOut = isSigOut_;
}

ge::graphStatus AvgPoolCommonBigKernelTiling::DoOpTiling()
{
    OP_LOGD(context_, "AvgPoolCommonBigKernelTiling DoOpTiling begin.");
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonBigKernelTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonBigKernelTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = 0;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPoolCommonBigKernelTiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    return ge::GRAPH_SUCCESS;
}

void AvgPoolCommonBigKernelTiling::DumpTilingInfo()
{
    AvgPool::AvgPoolBigKernelTilingData* tilingData =
        context_->GetTilingData<AvgPool::AvgPoolBigKernelTilingData>();
    std::ostringstream str;
    str << " hInDim:" << tilingData->hInDim;
    str << " wInDim:" << tilingData->wInDim;
    str << " hOutDim:" << tilingData->hOutDim;
    str << " wOutDim:" << tilingData->wOutDim;
    str << " kH:" << tilingData->kH;
    str << " kW:" << tilingData->kW;
    str << " sH:" << tilingData->sH;
    str << " sW:" << tilingData->sW;
    str << " tPad:" << tilingData->tPad;
    str << " bPad:" << tilingData->bPad;
    str << " lPad:" << tilingData->lPad;
    str << " rPad:" << tilingData->rPad;
    str << " divisorOverride:" << tilingData->divisorOverride;
    str << " countIncludePad:" << tilingData->countIncludePad;
    str << " blockFactor:" << tilingData->blockFactor;
    str << " blockTail:" << tilingData->blockTail;
    str << " totalIdx:" << tilingData->totalIdx;
    str << " coreNums:" << tilingData->coreNums;
    str << " maxCount:" << tilingData->maxCount;
    str << " isSigOut:" << tilingData->isSigOut;
    OP_LOGI(context_, "%s", str.str().c_str());
}

//////////////////////////////// AvgPoolBigKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolBigKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolPlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPoolBigKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolShapeAttrsInfo(context_, inputData_);
}

//////////////////////////////// AvgPoolV2BigKernelTiling /////////////////////////////////
ge::graphStatus AvgPoolV2BigKernelTiling::GetPlatformInfo()
{
    return GetAvgPoolV2PlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus AvgPoolV2BigKernelTiling::GetShapeAttrsInfo()
{
    return GetAvgPoolV2ShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("AvgPoolV2", AvgPoolV2BigKernelTiling, 2);
REGISTER_TILING_TEMPLATE("AvgPool", AvgPoolBigKernelTiling, 2);

}  // namespace optiling
