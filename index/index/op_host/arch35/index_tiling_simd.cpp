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
 * \file index_tiling_simd.cpp
 * \brief ac index tiling using simd
 */

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"
#include "index_tiling.h"
#include "index_tiling_simd.h"

using namespace AscendC;

namespace optiling {

static constexpr uint32_t SIMD_TILING_KEY = 3001;

static constexpr size_t X_INPUT_IDX = 0;
static constexpr size_t MASK_INPUT_IDX = 1;
static constexpr size_t INDICES_IDX = 3;
static constexpr size_t Y_OUTPUT_IDX = 0;
// static constexpr size_t INDEXED_SIZES_IDX = 1;
static constexpr uint32_t MAX_DIM = 8;
static constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint32_t>(16 * 1024 * 1024);
static constexpr int32_t INDICES_SIZE = 8192;
static constexpr int32_t BUFFER_NUM = 2;
static constexpr uint64_t SIMD_THRES = 256;
static constexpr int32_t NUM_TWO = 2;
static constexpr uint32_t MASK_DIM_NUM = 1;
static constexpr int32_t NUM_ZERO = 0;
static constexpr int32_t NUM_ONE = 1;
static constexpr uint32_t MERGE_OUTPUT_SHAPE_DIM = 3;

ge::graphStatus IndexTilingSimd::CheckShapeInfo()
{
    auto xInputShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    inputShapes_ = Ops::Base::EnsureNotScalar(xInputShape->GetStorageShape());
    auto maskInputShape = context_->GetInputShape(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskInputShape);
    maskShape_ = Ops::Base::EnsureNotScalar(maskInputShape->GetStorageShape());
    auto yOutputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputShape);
    auto outputShape = Ops::Base::EnsureNotScalar(yOutputShape->GetStorageShape());
    outputLength_ = outputShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::GetShapeDtypeSize()
{
    auto xInput = context_->GetInputDesc(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    inputDtypeSize_ = ge::GetSizeByDataType(xInput->GetDataType());

    uint64_t curIndexSize = 0;
    for (size_t i = 0; i < MAX_DIM; ++i) {
        auto curTensor = context_->GetDynamicInputTensor(INDICES_IDX, i);
        if (curTensor == nullptr) {
            // the num of dims that are indexed
            indexedDimNum_ = i;
            break;
        } else {
            auto curIndexShape = context_->GetDynamicInputShape(INDICES_IDX, i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
            auto indexShape = Ops::Base::EnsureNotScalar(curIndexShape->GetStorageShape());
            curIndexSize = indexShape.GetShapeSize();
            indexSize_ = curIndexSize;
        }
    }
    if (context_->GetDynamicInputTensor(INDICES_IDX, NUM_ZERO) != nullptr && indexedDimNum_ == 0) {
        indexedDimNum_ = MAX_DIM;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::GetShapeAttrsInfo()
{
    if (CheckShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeDtypeSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t maskIndices = 0;
    maskTensor_ = context_->GetInputTensor(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskTensor_);
    for (int64_t i = 0; i < maskShape_.GetDim(NUM_ZERO); i++) {
        if (maskTensor_->GetData<int64_t>()[i] == NUM_ONE) {
            maskIndices++;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool IndexTilingSimd::MargeInputAxis()
{
    int64_t mergeSize = NUM_ONE;
    uint32_t mergeIdx = NUM_ZERO;
    bool noIndex = false;
    for (uint32_t dim = 0; dim < static_cast<uint32_t>(inputShapes_.GetDimNum()); dim++) {
        if (dim >= static_cast<uint32_t>(maskShape_.GetDim(0)) || !maskTensor_->GetData<int64_t>()[dim]) {
            mergeSize *= inputShapes_[dim];
            noIndex = true;
        } else {
            if (noIndex) {
                mergeInputShape_[mergeIdx++] = static_cast<uint64_t>(mergeSize);
                mergeSize = 1;
                noIndex = false;
            }
            mergeInputIndexed_[mergeIdx] = 1;
            mergeInputShape_[mergeIdx++] = inputShapes_[dim];
        }
    }
    if (noIndex) {
        mergeInputShape_[mergeIdx++] = static_cast<uint64_t>(mergeSize);
    }
    mergeInputShapeDim_ = mergeIdx;
    return true;
}

bool IndexTilingSimd::IsSimd()
{
    if (!MargeInputAxis()) {
        OP_LOGE(context_->GetNodeName(), "merge input shape error!");
        return false;
    }
    if (!(mergeInputIndexed_[static_cast<int32_t>(mergeInputShapeDim_ - 1)] != static_cast<uint32_t>(1) &&
          mergeInputShape_[static_cast<int32_t>(mergeInputShapeDim_ - 1)] >= SIMD_THRES &&
          ubSize_ >= NUM_TWO * static_cast<uint64_t>(indexedDimNum_) * static_cast<uint64_t>(INDICES_SIZE) &&
          indexSize_ >= coreNum_ / static_cast<uint64_t>(NUM_TWO))) {
        return false;
    }
    isSimdTemplate_ = true;
    return true;
}

bool IndexTilingSimd::IsCapable()
{
    return IsSimd();
}

void IndexTilingSimd::CalcSimdTiling()
{
    int64_t blockFactor = static_cast<int64_t>(outputLength_) /
                          static_cast<int64_t>(mergeOutputShape_[mergeOutputShapeDim_ - 1]) /
                          static_cast<int64_t>(coreNum_);
    int64_t tailBlockFactor =
        static_cast<int64_t>(outputLength_) / static_cast<int64_t>(mergeOutputShape_[mergeOutputShapeDim_ - 1]) -
        blockFactor * static_cast<int64_t>(coreNum_);
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t ubAviable = (static_cast<int64_t>(ubSize_) - static_cast<int64_t>(indexedDimNum_) * INDICES_SIZE) /
                        ubBlockSize * ubBlockSize / static_cast<int64_t>(inputDtypeSize_) / BUFFER_NUM;
    needCoreNum_ = tailBlockFactor;
    if (blockFactor > 0) {
        needCoreNum_ = static_cast<int64_t>(coreNum_);
    }

    simdTilingData_.set_needCoreNum(needCoreNum_);
    simdTilingData_.set_perCoreElements(blockFactor);
    simdTilingData_.set_lastCoreElements(tailBlockFactor);
    simdTilingData_.set_maxElement(ubAviable);
    simdTilingData_.set_indiceUbSize(indexedDimNum_ * INDICES_SIZE);
    simdTilingData_.set_inputDtypeSize(inputDtypeSize_);
    simdTilingData_.set_indexedDimNum(indexedDimNum_);
    simdTilingData_.set_mergeInputShape(mergeInputShape_);
    simdTilingData_.set_mergeInputIndexed(mergeInputIndexed_);
    simdTilingData_.set_mergeInputShapeDim(mergeInputShapeDim_);
    simdTilingData_.set_mergeOutputShape(mergeOutputShape_);
    simdTilingData_.set_mergeOutToInput(mergeOutToInput_);
    simdTilingData_.set_indicesToInput(indicesToInput_);
    simdTilingData_.set_mergeOutputShapeDim(mergeOutputShapeDim_);
    simdTilingData_.set_isZeroOneZero(isZeroOneZero_);
    simdTilingData_.set_indexSize(indexSize_);
}

bool IndexTilingSimd::IsIndexContinue()
{
    int32_t firstIndexedDim = -1;
    int32_t lastIndexedDim = -1;
    for (int32_t dim = 0; dim < static_cast<int32_t>(maskShape_.GetDim(0)); dim++) {
        if (maskTensor_->GetData<int64_t>()[dim] == 1) {
            if (firstIndexedDim == -1) {
                firstIndexedDim = dim;
            }
            lastIndexedDim = dim;
        }
    }
    for (int32_t dim = firstIndexedDim; dim <= lastIndexedDim; dim++) {
        if (maskTensor_->GetData<int64_t>()[dim] == 0) {
            return false;
        }
    }
    return true;
}

/**
 * 合轴是对输入轴和输出轴进行合轴，要使用输出索引通过除法获取每个轴上的偏移，然后针对indices轴的偏移计算indices的值，与非indices轴偏移相加最终得到搬运位置。
 * marge axis
 * indices: [gather_size]
 * x:       [..., no_indices, indices, inner_size]
 * 判断indices是否连续，连续如下：
 * out:     [other_size, gather_size, inner_size]
 * 不连续，则：
 * out:     [gather_size, ..., other_size, ..., inner_size]
 */
ge::graphStatus IndexTilingSimd::MargeOutputAxis()
{
    bool isIndexContinue = IsIndexContinue();
    uint32_t indicesToInputDim = 0;
    if ((isIndexContinue && maskTensor_->GetData<int64_t>()[0]) || !isIndexContinue) {
        mergeOutputShape_[0] = indexSize_;
        uint32_t mergeOutDim = 1;
        for (uint32_t idx = 0; idx < mergeInputShapeDim_; idx++) {
            if (mergeInputIndexed_[idx] != static_cast<uint32_t>(NUM_ONE)) {
                mergeOutputShape_[mergeOutDim] = mergeInputShape_[idx];
                mergeOutToInput_[mergeOutDim++] = static_cast<int32_t>(idx);
            } else {
                indicesToInput_[indicesToInputDim++] = static_cast<int32_t>(idx);
            }
        }
        mergeOutputShapeDim_ = mergeOutDim;
    } else {
        mergeOutputShape_[0] = mergeInputShape_[0];
        mergeOutToInput_[0] = NUM_ZERO;
        mergeOutputShape_[1] = indexSize_;
        for (uint32_t idx = 0; idx < mergeInputShapeDim_; idx++) {
            if (mergeInputIndexed_[idx] == static_cast<uint32_t>(NUM_ONE)) {
                indicesToInput_[indicesToInputDim++] = static_cast<int32_t>(idx);
            }
        }
        mergeOutputShape_[NUM_TWO] = mergeInputShape_[static_cast<int32_t>(mergeInputShapeDim_) - 1];
        mergeOutToInput_[NUM_TWO] = static_cast<int32_t>(mergeInputShapeDim_ - 1);
        mergeOutputShapeDim_ = MERGE_OUTPUT_SHAPE_DIM;
        isZeroOneZero_ = 1;
    }

    uint64_t curShapeSum = 1;
    for (int32_t idx = static_cast<int32_t>(mergeOutputShapeDim_) - 2; idx > 0; idx--) {
        curShapeSum *= mergeOutputShape_[idx];
        mergeOutputShape_[idx] = curShapeSum;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::DoOpTiling()
{
    OP_CHECK_IF(
        MargeOutputAxis() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Marge axis error!"),
        return ge::GRAPH_FAILED);
    CalcSimdTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexTilingSimd::GetTilingKey() const
{
    return SIMD_TILING_KEY;
}

ge::graphStatus IndexTilingSimd::GetWorkspaceSize()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (simdTilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    simdTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(simdTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(Index, IndexTilingSimd, 1);
} // namespace optiling
