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
 * \file unsorted_segment_output_fullload_tiling.cpp
 * \brief unsorted_segment_output_fullload_tiling
 */
#include "unsorted_segment_output_fullload_tiling.h"

namespace optiling {
using namespace UnsortedSegmentMax;
static constexpr uint32_t ROW_NUM = 16;
static constexpr uint32_t MAX_LAST_DIM_RANG = 1024;
static constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(32 * 1024);
static constexpr uint64_t BUFFER_ADD_NUM = 2;
static constexpr uint64_t UB_MIN_FACTOR = 2048;
static constexpr uint64_t SIMT_BLOCK_DIM_MAX = 2048;
static constexpr uint64_t PARALLEL_NUM_CAP = 1024;
static constexpr uint64_t ROW_UB_LOOPS_TARGET = 1024;

static uint32_t FloorPow2(uint32_t x)
{
    if (x == 0U) {
        return 0U;
    }
    uint32_t r = 1U;
    while ((r << 1U) <= x && (r << 1U) != 0U) {
        r <<= 1U;
    }
    return r;
}

bool UnsortedSegmentOutFlTiling::IsCapable()
{
    if ((IsSupportDtype() && IsSupportSize()) || dataShapeSize_ == 0UL) {
        return true;
    }
    return false;
}

bool UnsortedSegmentOutFlTiling::IsSupportDtype()
{
    if (dataType_ != ge::DT_INT64 && dataType_ != ge::DT_UINT64 && dataType_ != ge::DT_UINT32) {
        return true;
    }
    return false;
}

bool UnsortedSegmentOutFlTiling::IsSupportSize()
{
    uint32_t lastDimRange = MAX_LAST_DIM_RANG / ROW_NUM;
    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t xBufferSize = ROW_NUM * innerDim_ * dataTypeBytes_ + ubBlockSize_;
    uint64_t indexBufferSize = ROW_NUM * idTypeBytes_ + ubBlockSize_;
    uint64_t tmpBufferSize = Ops::Base::CeilAlign(outSize * dataTypeBytes_, ubBlockSize_) * ROW_NUM;
    uint64_t needUb = xBufferSize * BUFFER_ADD_NUM + indexBufferSize * BUFFER_ADD_NUM + tmpBufferSize;
    ubSize_ = ubSize_ - DCACHE_SIZE;
    if (needUb < ubSize_ && innerDimAlign_ <= lastDimRange) {
        return true;
    }
    return false;
}

ge::graphStatus UnsortedSegmentOutFlTiling::DoOpTiling()
{
    UbAddBranch();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentOutFlTiling::UbAddBranch()
{
    if (dataShapeSize_ == 0UL) {
        uint64_t outSize = outputOuterDim_ * innerDim_;
        uint64_t normBlockData = Ops::Base::CeilDiv(outSize, totalCoreNum_);
        normBlockData = std::max(normBlockData, UB_MIN_FACTOR / dataTypeBytes_);
        usedCoreNum_ = Ops::Base::CeilDiv(outSize, normBlockData);
        return ge::GRAPH_SUCCESS;
    }
    maxIndexNum_ = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
    if (1UL == maxIndexNum_) {
        usedCoreNum_ = std::min(inputOuterDim_, totalCoreNum_);
    } else {
        usedCoreNum_ = std::min(Ops::Base::CeilDiv(inputOuterDim_, maxIndexNum_), totalCoreNum_);
    }
    uint64_t oneRowinnerDimSize = innerDim_ * dataTypeBytes_;

    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t oneRowOutNumAlignBytes = Ops::Base::CeilAlign(outSize * dataTypeBytes_, ubBlockSize_);
    oneRowOutNumAlign_ = oneRowOutNumAlignBytes / dataTypeBytes_;
    uint64_t accStrideBytes = oneRowOutNumAlignBytes;
    if ((accStrideBytes / ubBlockSize_) % 2UL == 0UL) {
        accStrideBytes += ubBlockSize_;
    }
    accStride_ = accStrideBytes / dataTypeBytes_;

    uint32_t pThreadRaw = innerDim_ > 0UL ? static_cast<uint32_t>(SIMT_BLOCK_DIM_MAX / innerDim_) : 1U;
    uint32_t pThread = FloorPow2(std::min(pThreadRaw, static_cast<uint32_t>(PARALLEL_NUM_CAP)));

    uint32_t pRow = FloorPow2(static_cast<uint32_t>(std::max(maxIndexNum_, static_cast<uint64_t>(ROW_NUM))));
    uint32_t pCap = std::min(pThread, pRow);
    if (pCap < ROW_NUM) {
        pCap = ROW_NUM;
    }

    uint32_t chosenP = ROW_NUM;
    uint64_t chosenRowUb = 0UL;
    constexpr uint32_t P_LOOP1_CAP = 128U;
    uint32_t pCapLoop1 = std::min(pCap, P_LOOP1_CAP);
    for (uint32_t p = pCapLoop1; p >= ROW_NUM; p >>= 1U) {
        uint64_t tmpBufSize = accStrideBytes * p;
        if (tmpBufSize >= ubSize_) {
            continue;
        }
        uint64_t remainUbSize = (ubSize_ - tmpBufSize) / BUFFER_ADD_NUM;
        uint64_t rowUb = 0UL;
        if (remainUbSize > ubBlockSize_ + ubBlockSize_) {
            rowUb = (remainUbSize - ubBlockSize_ - ubBlockSize_) / (oneRowinnerDimSize + idTypeBytes_);
        }
        if (rowUb >= maxIndexNum_) {
            chosenP = p;
            chosenRowUb = maxIndexNum_;
            break;
        }
    }
    if (chosenRowUb == 0UL) {
        for (uint32_t p = pCap; p >= ROW_NUM; p >>= 1U) {
            uint64_t tmpBufSize = accStrideBytes * p;
            if (tmpBufSize >= ubSize_) {
                continue;
            }
            uint64_t remainUbSize = (ubSize_ - tmpBufSize) / BUFFER_ADD_NUM;
            uint64_t rowUb = 0UL;
            if (remainUbSize > ubBlockSize_ + ubBlockSize_) {
                rowUb = (remainUbSize - ubBlockSize_ - ubBlockSize_) / (oneRowinnerDimSize + idTypeBytes_);
            }
            if (rowUb > maxIndexNum_) {
                rowUb = maxIndexNum_;
            }
            if (rowUb >= ROW_UB_LOOPS_TARGET || p == ROW_NUM) {
                chosenP = p;
                chosenRowUb = rowUb;
                break;
            }
        }
    }
    if (chosenRowUb == 0UL) {
        chosenRowUb = 1UL;
    }
    parallelNum_ = chosenP;
    rowNumUb_ = chosenRowUb;
    oneCoreUbLoopTimes_ = (maxIndexNum_ + rowNumUb_ - 1UL) / rowNumUb_;
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentOutFlTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_OUT_FL, USS_CAST_NONE);
    return tilingKey;
}

void UnsortedSegmentOutFlTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentOutFlTilingData*
        tilingData = context_->GetTilingData<UnsortedSegment::UnsortedSegmentOutFlTilingData>();

    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->maxIndexNum = maxIndexNum_;
    tilingData->oneCoreUbLoopTimes = oneCoreUbLoopTimes_;
    tilingData->rowNumUb = rowNumUb_;
    tilingData->parallelNum = parallelNum_;
    tilingData->oneRowOutNumAlign = oneRowOutNumAlign_;
    tilingData->accStride = accStride_;
}

ge::graphStatus UnsortedSegmentOutFlTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentOutFlTiling::DumpTilingInfo()
{
    UnsortedSegment::UnsortedSegmentOutFlTilingData*
        tilingData = context_->GetTilingData<UnsortedSegment::UnsortedSegmentOutFlTilingData>();

    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << tilingData->inputOuterDim;
    info << ", outputOuterDim: " << tilingData->outputOuterDim;
    info << ", innerDim: " << tilingData->innerDim;
    info << ", maxIndexNum: " << tilingData->maxIndexNum;
    info << ", oneCoreUbLoopTimes: " << tilingData->oneCoreUbLoopTimes;
    info << ", rowNumUb: " << tilingData->rowNumUb;
    info << ", parallelNum: " << tilingData->parallelNum;
    info << ", oneRowOutNumAlign: " << tilingData->oneRowOutNumAlign;
    info << ", accStride: " << tilingData->accStride;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling
