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
 * \file unsorted_segment_prod_output_fullload_ws_merge_tiling.cpp
 * \brief unsorted_segment_prod_output_fullload_ws_merge_tiling
 */

#include "unsorted_segment_prod_output_fullload_ws_merge_tiling.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

using namespace AscendC;
using namespace UnsortedSegmentProd;

namespace optiling {

namespace {
constexpr uint64_t OFW_SYS_WORKSPACE = 16UL * 1024 * 1024;
constexpr uint64_t OFW_WORKSPACE_LIMIT = 64UL * 1024 * 1024;
constexpr uint64_t OFW_MERGE_CHUNK_HOST = 4096;
constexpr uint64_t OFW_MIN_OUT_SIZE = 128;
constexpr uint64_t OFW_OUTFL_ECONOMIC_FACTOR = 2;
constexpr uint64_t OFW_ROW_NUM = 16;
constexpr uint64_t OFW_MERGE_BUF_NUM = 3UL;
} // namespace

bool UnsortedSegmentProdOutFlWsMergeTiling::IsCapable()
{
    if (!UnsortedSegmentOutFlTiling::IsCapable()) {
        return false;
    }
    if (dataShapeSize_ == 0UL || inputOuterDim_ == 0UL) {
        return false;
    }
    uint64_t outputSize = outputOuterDim_ * innerDim_;
    if (inputOuterDim_ <= outputSize * totalCoreNum_ * OFW_OUTFL_ECONOMIC_FACTOR) {
        return false;
    }
    if (outputSize < OFW_MIN_OUT_SIZE) {
        return false;
    }
    uint64_t tmpBytes = Ops::Base::CeilAlign(outputSize * dataTypeBytes_, ubBlockSize_) * OFW_ROW_NUM;
    uint64_t xBytes = (OFW_ROW_NUM * innerDim_ * dataTypeBytes_ + ubBlockSize_) * 2UL;
    uint64_t idxBytes = (OFW_ROW_NUM * idTypeBytes_ + ubBlockSize_) * 2UL;
    uint64_t mergeBytes = OFW_MERGE_BUF_NUM * OFW_MERGE_CHUNK_HOST * dataTypeBytes_;
    if (xBytes + idxBytes + tmpBytes + mergeBytes >= ubSize_) {
        return false;
    }
    uint64_t wsStrideElems = Ops::Base::CeilAlign(outputSize * dataTypeBytes_, ubBlockSize_) / dataTypeBytes_;
    if (totalCoreNum_ * wsStrideElems * dataTypeBytes_ > OFW_WORKSPACE_LIMIT) {
        return false;
    }
    return true;
}

ge::graphStatus UnsortedSegmentProdOutFlWsMergeTiling::UbAddBranchFixedP()
{
    constexpr uint64_t BUFFER_ADD_NUM = 2;

    maxIndexNum_ = Ops::Base::CeilDiv(inputOuterDim_, totalCoreNum_);
    if (1UL == maxIndexNum_) {
        usedCoreNum_ = std::min(inputOuterDim_, totalCoreNum_);
    } else {
        usedCoreNum_ = std::min(Ops::Base::CeilDiv(inputOuterDim_, maxIndexNum_), totalCoreNum_);
    }

    uint64_t outSize = outputOuterDim_ * innerDim_;
    uint64_t oneRowOutNumAlignBytes = Ops::Base::CeilAlign(outSize * dataTypeBytes_, ubBlockSize_);
    oneRowOutNumAlign_ = oneRowOutNumAlignBytes / dataTypeBytes_;
    accStride_ = oneRowOutNumAlign_;
    parallelNum_ = OFW_ROW_NUM;

    uint64_t tmpBufSize = oneRowOutNumAlignBytes * parallelNum_;
    uint64_t remainUbSize = (ubSize_ > tmpBufSize) ? (ubSize_ - tmpBufSize) / BUFFER_ADD_NUM : 0UL;
    uint64_t rowUb = 0UL;
    if (remainUbSize > ubBlockSize_ * 2UL) {
        rowUb = (remainUbSize - ubBlockSize_ * 2UL) / (innerDim_ * dataTypeBytes_ + idTypeBytes_);
    }
    if (rowUb > maxIndexNum_) {
        rowUb = maxIndexNum_;
    }
    if (rowUb == 0UL) {
        rowUb = 1UL;
    }
    rowNumUb_ = rowUb;
    oneCoreUbLoopTimes_ = (maxIndexNum_ + rowNumUb_ - 1UL) / rowNumUb_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentProdOutFlWsMergeTiling::DoOpTiling()
{
    UbAddBranchFixedP();
    UnsortedSegmentOutFlTiling::SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentProdOutFlWsMergeTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(USS_TEMPLATE_OUT_FL_WS_MERGE, USS_CAST_NONE);
    return tilingKey;
}

ge::graphStatus UnsortedSegmentProdOutFlWsMergeTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    uint64_t outSize = outputOuterDim_ * innerDim_;
    wsStride_ = Ops::Base::CeilAlign(outSize * dataTypeBytes_, ubBlockSize_) / dataTypeBytes_;
    workspaces[0] = OFW_SYS_WORKSPACE + usedCoreNum_ * wsStride_ * dataTypeBytes_;
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentProdOutFlWsMergeTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", outputOuterDim: " << outputOuterDim_;
    info << ", innerDim: " << innerDim_;
    info << ", maxIndexNum: " << maxIndexNum_;
    info << ", oneCoreUbLoopTimes: " << oneCoreUbLoopTimes_;
    info << ", rowNumUb: " << rowNumUb_;
    info << ", parallelNum(fixed): " << parallelNum_;
    info << ", wsStride: " << wsStride_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("UnsortedSegmentProd", UnsortedSegmentProdOutFlWsMergeTiling, 8);

} // namespace optiling
