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

#include <limits>
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

size_t IndexPutWithSortV2SIMDTiling::FindFirstOneIdx()
{
    for (size_t i = 0; i < selfDimNum_; ++i) {
        if (indexedSizes_[i] == 1) {
            return i;
        }
    }
    return selfDimNum_;
}

bool IndexPutWithSortV2SIMDTiling::IsCapable()
{
    constexpr int64_t MAX_SAFE_MULTIPLY = std::numeric_limits<int64_t>::max();
    int64_t dtypeSize = ge::GetSizeByDataType(xDataType_);
    bool requiresNonIndexed = false;
    if (nonIndexedDimSize_ > 0 && dtypeSize > 0 && 
        nonIndexedDimSize_ <= MAX_SAFE_MULTIPLY / dtypeSize) {
        requiresNonIndexed = nonIndexedDimSize_ * dtypeSize > SPLIT_LIMIT;
    }
    
    if (CheckIndexedSizesPattern() && requiresNonIndexed) {
        size_t firstOneIdx = FindFirstOneIdx();

        size_t newDimNum = selfDimNum_ - firstOneIdx;
        for (size_t i = 0; i < newDimNum; ++i) {
            if (firstOneIdx + i >= MAX_DIM_NUM || i >= MAX_DIM_NUM) {
                OP_LOGI(context_->GetNodeName(), "IsCapable: array index out of bounds");
                return false;
            }
            indexedSizes_[i] = indexedSizes_[firstOneIdx + i];
            selfDims_[i] = selfDims_[firstOneIdx + i];
        }
        
        selfDimNum_ = static_cast<uint32_t>(newDimNum);
        
        indexedDimNum_ = 0;
        nonIndexedDimNum_ = 0;
        nonIndexedDimSize_ = 1;
        for (size_t i = 0; i < selfDimNum_; ++i) {
            if (indexedSizes_[i] == 0) {
                nonIdxedDims_[nonIndexedDimNum_] = selfDims_[i];
                nonIndexedDimNum_++;
                if (nonIndexedDimSize_ > 0 && selfDims_[i] > 0 &&
                    nonIndexedDimSize_ <= MAX_SAFE_MULTIPLY / selfDims_[i]) {
                    nonIndexedDimSize_ *= selfDims_[i];
                } else {
                    OP_LOGI(context_->GetNodeName(), "IsCapable: nonIndexedDimSize_ multiplication overflow detected");
                    return false;
                }
            } else {
                indexedDimNum_++;
            }
        }
        
        indexed0_ = indexedSizes_[0];
        isContinous_ = true;
        
        OP_LOGI(context_->GetNodeName(), 
                "SIMD adapted: firstOneIdx=%zu, newDimNum=%u, indexedDimNum=%u, nonIndexedDimNum=%u, "
                "indexed0_=%ld, isContinous_=%d",
                firstOneIdx, selfDimNum_, indexedDimNum_, nonIndexedDimNum_,
                indexed0_, static_cast<int>(isContinous_));
        
        return true;
    }
    
    OP_LOGI(context_->GetNodeName(), 
            "IndexPutWithSortV2SIMDTiling IsCapable isContinous_: %ld, nonIndexedDimSize_: %ld", 
            static_cast<int64_t>(isContinous_), nonIndexedDimSize_);
    isContinous_ = (isContinous_ && (indexed0_ == 1));
    return isContinous_ && requiresNonIndexed;
}

// [001100]
bool IndexPutWithSortV2SIMDTiling::CheckIndexedSizesPattern()
{
    auto const inputShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto const inShapeVal = inputShape->GetStorageShape();
    for (size_t i = 0; i < selfDimNum_; ++i) {
        inputShapes_[i] = inShapeVal.GetDim(i);
    }

    if (selfDimNum_ == 0 || selfDimNum_ > MAX_DIM_NUM) {
        OP_LOGI(context_->GetNodeName(), "CheckIndexedSizesPattern: invalid selfDimNum_=%u", selfDimNum_);
        return false;
    }

    size_t firstOneIdx = FindFirstOneIdx();

    if (firstOneIdx == selfDimNum_) {
        return false;
    }

    for (size_t i = 0; i < firstOneIdx; ++i) {
        if (indexedSizes_[i] != 0) {
            return false;
        }
    }

    for (size_t i = firstOneIdx; i < selfDimNum_; ++i) {
        if (indexedSizes_[i] != 1) {
            for (size_t j = i + 1; j < selfDimNum_; ++j) {
                if (indexedSizes_[j] != 0) {
                    return false;
                }
            }
            break;
        }
    }

    if (indexedSizes_[selfDimNum_ - 1] != 0) {
        return false;
    }

    for (size_t i = 0; i < firstOneIdx; ++i) {
        if (inputShapes_[i] != 1) {
            return false;
        }
    }

    OP_LOGI(context_->GetNodeName(), "CheckIndexedSizesPattern: pattern check passed");
    return true;
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