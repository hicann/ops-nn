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
* \file index_put_with_sort_v2_tiling_arch35.cpp
* \brief IndexPutWithSortV2 regbase tiling file
*/

#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "index_put_with_sort_v2_simd_tiling_arch35.h"
#include "index/index_put_with_sort_v2/op_kernel/arch35/index_put_with_sort_v2_struct.h"

namespace optiling
{
using namespace Ops::NN::OpTiling;

constexpr int64_t INPUT_NUM = 4;
constexpr int64_t OUTPUT_NUM = 1;
constexpr int64_t ATTR_INDEX_0 = 0;
constexpr int64_t ATTR_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_0 = 0;
constexpr int64_t INPUT_INDEX_1 = 1;
constexpr int64_t INPUT_INDEX_2 = 2;
constexpr int64_t INPUT_INDEX_3 = 3;
constexpr int64_t INPUT_INDEX_4 = 4;
constexpr uint32_t DCACHE_SIZE = 32768;
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216;
constexpr int64_t UB_BLOCK = 32;
constexpr int64_t COL_ALING = 512;
constexpr int32_t SPLIT_LIMIT = 256;
constexpr int64_t MIN_UB_FOR_INDICES = 8*1024; // 8K;

bool IndexPutWithSortV2SIMDTiling::IsCapable()
{
    OP_LOGI(context_->GetNodeName(), "IndexPutWithSortV2SIMDTiling IsCapable isContinous_: %ld, nonIndexedDimSize_: %ld", static_cast<int64_t>(isContinous_), nonIndexedDimSize_);
    isContinous_ = (isContinous_ && (indexed0_ == 1));
    return isContinous_ && (nonIndexedDimSize_ * ge::GetSizeByDataType(xDataType_) > SPLIT_LIMIT);
}


uint64_t IndexPutWithSortV2SIMDTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(accumulate_, nonIndexedDimSize_ == 1, indexedBlockMode_, isCast_, true);
}

ge::graphStatus IndexPutWithSortV2SIMDTiling::PostTiling()
{
    context_->SetBlockDim(rowUseCoreNum_ * colUseCoreNum_);
    context_->SetScheduleMode(1);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    currentWorkspace[0] = workspaceSize_ + ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutWithSortV2SIMDTiling::GetWorkspaceSize()
{
    workspaceSize_ = Ops::Base::CeilAlign(nonIndexedDimSize_ * xCastDtypeSize_, COL_ALING) * B2_SIZE * rowUseCoreNum_ +
    rowUseCoreNum_ * B2_SIZE * CACHELINE_SIZE;
    return ge::GRAPH_SUCCESS;
}


void IndexPutWithSortV2SIMDTiling::LogTilingResult()
{
    OP_LOGI(context_->GetNodeName(), "indexedDimSize_: %ld, nonIndexedDimSize_: %ld", indexedDimSize_, nonIndexedDimSize_);
}

void IndexPutWithSortV2SIMDTiling::SetTilingData()
{
    IndexPutWithSortV2SimdTilingData* tilingData =
    context_->GetTilingData<IndexPutWithSortV2SimdTilingData>();
    tilingData->nonIndexedDimNum = static_cast<int64_t>(nonIndexedDimNum_);
    tilingData->indexedDimSize = indexedDimSize_;
    tilingData->nonIndexedDimSize = nonIndexedDimSize_;

    tilingData->indicesFactor = indicesFactor_;
    tilingData->ubFactor = ubFactor_;
    tilingData->rowBlockFactor = rowBlockFactor_;
    tilingData->rowUseCoreNum = rowUseCoreNum_;
    tilingData->rowTailBlockFactor = rowTailBlockFactor_;
    tilingData->colBlockFactor = colBlockFactor_;
    tilingData->colUseCoreNum = colUseCoreNum_;
    tilingData->colTailBlockFactor = colTailBlockFactor_;
}

void IndexPutWithSortV2SIMDTiling::DoBlockTiling()
{
    rowBlockFactor_ = Ops::Base::CeilDiv(indexedDimSize_, aivCoreNum_);
    rowUseCoreNum_ = Ops::Base::CeilDiv(indexedDimSize_, rowBlockFactor_);
    rowTailBlockFactor_ = indexedDimSize_ - (rowUseCoreNum_ - 1) * rowBlockFactor_;
    colBlockFactor_ = nonIndexedDimSize_;
    colUseCoreNum_ = 1;
    colTailBlockFactor_ = nonIndexedDimSize_ - (colUseCoreNum_ - 1) * colBlockFactor_;
    if (rowUseCoreNum_ < aivCoreNum_ / B2_SIZE) {
        int64_t tmpColBlockNum = aivCoreNum_ / rowUseCoreNum_;
        colBlockFactor_ = Ops::Base::CeilDiv(nonIndexedDimSize_, tmpColBlockNum);
        colBlockFactor_ = Ops::Base::CeilAlign(colBlockFactor_, COL_ALING / ge::GetSizeByDataType(xDataType_));
        colUseCoreNum_ = Ops::Base::CeilDiv(nonIndexedDimSize_, colBlockFactor_);
        colTailBlockFactor_ = nonIndexedDimSize_ - (colUseCoreNum_ - 1) * colBlockFactor_;
    }
}

void IndexPutWithSortV2SIMDTiling::DoUbTiling()
{
    auto xTypeSize = ge::GetSizeByDataType(xDataType_);
    int64_t availableUb = Ops::Base::FloorAlign(maxUbSize_ / B2_SIZE, static_cast<uint64_t>(UB_BLOCK));  // double buff
    inOutUb_ = Ops::Base::CeilAlign(colBlockFactor_ * xTypeSize, UB_BLOCK);   // 一行输入/输出
    xCastDtypeSize_ = xTypeSize;
    int64_t resUb = availableUb - (inOutUb_ * B2_SIZE); // input + output
    if ((xDataType_ == ge::DT_FLOAT16 || xDataType_ == ge::DT_BF16) && accumulate_) {
        isCast_ = true;
        xCastDtypeSize_ = B4_SIZE;
        int64_t castUb = inOutUb_ * (B4_SIZE / xTypeSize);
        resUb = availableUb - (inOutUb_ + castUb * B2_SIZE); // input + output + cast ， 
    }
    
    if (resUb >= MIN_UB_FOR_INDICES) {
        indicesFactor_ = Ops::Base::FloorAlign(resUb / static_cast<int64_t>(B2_SIZE), UB_BLOCK) / indicesTypeSize_;     // indices + pos    100   98
        ubFactor_ = colBlockFactor_;
    } else {
        indicesFactor_ = MIN_UB_FOR_INDICES / B2_SIZE / indicesTypeSize_;
        ubFactor_ = Ops::Base::FloorAlign((availableUb - MIN_UB_FOR_INDICES) / static_cast<int64_t>(B2_SIZE), UB_BLOCK) / xCastDtypeSize_; 
    }
}

ge::graphStatus IndexPutWithSortV2SIMDTiling::DoOpTiling()
{
    OP_LOGI(context_->GetNodeName(), "IndexPutWithSortV2SIMDTiling DoOpTiling");
    DoBlockTiling();
    DoUbTiling();
    SetTilingData();
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(IndexPutWithSortV2, IndexPutWithSortV2SIMDTiling, 10);

}  // namespace optiling