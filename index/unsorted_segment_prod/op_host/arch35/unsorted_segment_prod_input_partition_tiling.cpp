/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_prod_input_partition_tiling.cpp
 * \brief unsorted_segment_prod_input_partition_tiling
 */

#include "unsorted_segment_prod_input_partition_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

namespace {
constexpr uint64_t IP_MIN_CORE_GAIN = 4;
constexpr uint64_t IP_WIDE_CORE_GAIN = 16;
constexpr uint64_t IP_MIN_INPUT_ROWS = 1024;
constexpr uint64_t IP_MERGE_CHUNK_MAX = 4096;
constexpr uint64_t IP_UB_RESERVE = 8 * 1024;
constexpr uint64_t IP_SYS_WORKSPACE = 16UL * 1024 * 1024;
constexpr uint64_t IP_WORKSPACE_LIMIT = 64UL * 1024 * 1024;
constexpr uint64_t IP_FLAG_STRIDE = 16;
} // namespace

bool UnsortedSegmentProdInputPartTiling::IsCapable()
{
    if (dataType_ == ge::DT_INT64 || dataType_ == ge::DT_UINT64) {
        return false;
    }
    if (dataShapeSize_ == 0UL || innerDim_ == 0UL || outputOuterDim_ == 0UL || inputOuterDim_ == 0UL) {
        return false;
    }
    uint64_t colNormBlock = Ops::Base::CeilAlign(Ops::Base::CeilDiv(innerDim_, totalCoreNum_),
                                                 ubBlockSize_ / dataTypeBytes_);
    uint64_t colUsedCore = Ops::Base::CeilDiv(innerDim_, colNormBlock);
    if (colUsedCore >= IP_WIDE_CORE_GAIN) {
        return false;
    }
    if (inputOuterDim_ < IP_MIN_INPUT_ROWS) {
        return false;
    }
    innerAlign_ = Ops::Base::CeilAlign(innerDim_ * dataTypeBytes_, ubBlockSize_) / dataTypeBytes_;
    ySize_ = outputOuterDim_ * innerAlign_;
    uint64_t yBytes = ySize_ * dataTypeBytes_;
    uint64_t xBytes = innerAlign_ * dataTypeBytes_ * 2UL;
    uint64_t idsBytes = (idTypeBytes_ + ubBlockSize_) * 2UL;
    uint64_t mergeBytes = IP_MERGE_CHUNK_MAX * dataTypeBytes_ * 2UL;
    if (yBytes + xBytes + idsBytes + mergeBytes + IP_UB_RESERVE >= ubSize_) {
        return false;
    }
    uint64_t rows = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
    uint64_t partCore = Ops::Base::CeilDiv(inputOuterDim_, rows);
    partCore = std::min(partCore, totalCoreNum_);
    if (partCore <= colUsedCore) {
        return false;
    }
    if (partCore * yBytes > IP_WORKSPACE_LIMIT) {
        return false;
    }
    return true;
}

ge::graphStatus UnsortedSegmentProdInputPartTiling::DoOpTiling()
{
    innerAlign_ = Ops::Base::CeilAlign(innerDim_ * dataTypeBytes_, ubBlockSize_) / dataTypeBytes_;
    ySize_ = outputOuterDim_ * innerAlign_;

    normRowNum_ = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
    partCoreNum_ = Ops::Base::CeilDiv(inputOuterDim_, normRowNum_);
    partCoreNum_ = std::min(partCoreNum_, totalCoreNum_);

    uint64_t yBytes = ySize_ * dataTypeBytes_;
    uint64_t mergeBytes = IP_MERGE_CHUNK_MAX * dataTypeBytes_ * 2UL;
    uint64_t avail = (ubSize_ > yBytes + mergeBytes + IP_UB_RESERVE) ? (ubSize_ - yBytes - mergeBytes - IP_UB_RESERVE) :
                                                                       0UL;
    uint64_t perRow = innerAlign_ * dataTypeBytes_ * 2UL + idTypeBytes_ * 2UL;
    baseS_ = (perRow > 0UL) ? (avail / perRow) : 1UL;
    baseS_ = std::max(baseS_, static_cast<uint64_t>(1));
    baseS_ = std::min(baseS_, normRowNum_);

    uint64_t rowsPerCore = Ops::Base::CeilDiv(outputOuterDim_, totalCoreNum_);
    mergeNormNum_ = rowsPerCore * innerAlign_;
    uint64_t chunkRows = (innerAlign_ > 0UL) ? (IP_MERGE_CHUNK_MAX / innerAlign_) : 1UL;
    chunkRows = std::max(chunkRows, static_cast<uint64_t>(1));
    chunkRows = std::min(chunkRows, rowsPerCore);
    mergeChunk_ = chunkRows * innerAlign_;

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentProdInputPartTiling::SetTilingData()
{
    UnsortedSegment::UnsortedSegmentProdInputPartTilingData*
        tilingData = context_->GetTilingData<UnsortedSegment::UnsortedSegmentProdInputPartTilingData>();
    tilingData->inputOuterDim = inputOuterDim_;
    tilingData->outputOuterDim = outputOuterDim_;
    tilingData->innerDim = innerDim_;
    tilingData->normRowNum = normRowNum_;
    tilingData->baseS = baseS_;
    tilingData->partCoreNum = partCoreNum_;
    tilingData->mergeNormNum = mergeNormNum_;
    tilingData->mergeChunk = mergeChunk_;
}

uint64_t UnsortedSegmentProdInputPartTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_INPUT_PART, USS_CAST_NONE);
    return tilingKey;
}

ge::graphStatus UnsortedSegmentProdInputPartTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    uint64_t usrSize = partCoreNum_ * ySize_ * dataTypeBytes_;
    usrSize += totalCoreNum_ * IP_FLAG_STRIDE * sizeof(int32_t);
    workspaces[0] = IP_SYS_WORKSPACE + usrSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentProdInputPartTiling::PostTiling()
{
    context_->SetBlockDim(totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentProdInputPartTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", outputOuterDim: " << outputOuterDim_;
    info << ", innerDim: " << innerDim_;
    info << ", innerAlign: " << innerAlign_;
    info << ", normRowNum: " << normRowNum_;
    info << ", baseS: " << baseS_;
    info << ", partCoreNum: " << partCoreNum_;
    info << ", mergeNormNum: " << mergeNormNum_;
    info << ", mergeChunk: " << mergeChunk_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdInputPartTiling, 15);

} // namespace optiling
