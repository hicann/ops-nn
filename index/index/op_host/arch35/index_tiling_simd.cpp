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
 * \brief SIMD tiling implementation for Index operator
 */

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/index_tiling_key.h"
#include "index_tiling.h"
#include "index_tiling_simd.h"

using namespace Index;
namespace optiling {
static constexpr size_t X_INPUT_IDX = 0;
static constexpr size_t MASK_INPUT_IDX = 1;
static constexpr size_t INDICES_IDX = 3;
static constexpr size_t Y_OUTPUT_IDX = 0;
static constexpr uint32_t MAX_DIM = 8;
static constexpr int32_t INDICES_SIZE = 8192;
static constexpr int32_t BUFFER_NUM = 2;
static constexpr uint64_t SIMD_THRES = 256;
static constexpr int32_t NUM_TWO = 2;
static constexpr int32_t NUM_ZERO = 0;
static constexpr int32_t NUM_ONE = 1;
static constexpr int32_t NUM_EIGHT = 8;
static constexpr uint32_t MERGE_OUTPUT_SHAPE_DIM = 3;

ge::graphStatus IndexTilingSimd::CheckShapeInfo()
{
    auto xInputShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    inputShapes_ = Ops::NN::OpTiling::EnsureNotScalar(xInputShape->GetStorageShape());
    auto maskInputShape = context_->GetInputShape(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskInputShape);
    maskShape_ = Ops::NN::OpTiling::EnsureNotScalar(maskInputShape->GetStorageShape());
    auto yOutputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputShape);
    auto outputShape = Ops::NN::OpTiling::EnsureNotScalar(yOutputShape->GetStorageShape());
    outputLength_ = outputShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::GetShapeDtypeSize()
{
    auto xInput = context_->GetInputDesc(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    inputDtypeSize_ = ge::GetSizeByDataType(xInput->GetDataType());

    for (size_t i = 0; i < MAX_DIM; ++i) {
        auto curTensor = context_->GetDynamicInputTensor(INDICES_IDX, i);
        if (curTensor == nullptr) {
            // the num of dims that are indexed
            indexedDimNum_ = i;
            break;
        } else {
            auto curIndexShape = context_->GetDynamicInputShape(INDICES_IDX, i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
            auto indexShape = Ops::NN::OpTiling::EnsureNotScalar(curIndexShape->GetStorageShape());
            indexSize_ = indexShape.GetShapeSize();
        }
    }
    if (context_->GetDynamicInputTensor(INDICES_IDX, NUM_ZERO) != nullptr && indexedDimNum_ == 0) {
        indexedDimNum_ = MAX_DIM;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexTilingSimd::GetShapeAttrsInfo()
{
    // 获取 tiling 数据
    simdTilingData_ = context_->GetTilingData<IndexSimdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, simdTilingData_);
    OP_CHECK_IF(
        memset_s(simdTilingData_, sizeof(IndexSimdTilingData), 0, sizeof(IndexSimdTilingData)) != EOK,
        OP_LOGE(context_->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    if (CheckShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeDtypeSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    maskTensor_ = context_->GetInputTensor(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskTensor_);
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

bool IndexTilingSimd::IsCapable()
{
    if (!MargeInputAxis()) {
        OP_LOGE(context_->GetNodeName(), "merge input shape error!");
        return false;
    }
    if (mergeInputShapeDim_ == 0) {
        return false;
    }
    if (!(mergeInputIndexed_[static_cast<int32_t>(mergeInputShapeDim_ - 1)] != static_cast<uint32_t>(1) &&
          mergeInputShape_[static_cast<int32_t>(mergeInputShapeDim_ - 1)] >= SIMD_THRES &&
          ubSize_ >= NUM_TWO * static_cast<uint64_t>(indexedDimNum_) * static_cast<uint64_t>(INDICES_SIZE) &&
          indexSize_ >= coreNum_ / static_cast<uint64_t>(NUM_TWO))) {
        return false;
    }
    return true;
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
    int64_t ubAvailable = (static_cast<int64_t>(ubSize_) - static_cast<int64_t>(indexedDimNum_) * INDICES_SIZE) /
                          ubBlockSize * ubBlockSize / static_cast<int64_t>(inputDtypeSize_) / BUFFER_NUM;
    needCoreNum_ = tailBlockFactor;
    if (blockFactor > 0) {
        needCoreNum_ = static_cast<int64_t>(coreNum_);
    }

    // simdTilingData_ 已经在 GetShapeAttrsInfo 中初始化
    simdTilingData_->needCoreNum = needCoreNum_;
    simdTilingData_->perCoreElements = blockFactor;
    simdTilingData_->lastCoreElements = tailBlockFactor;
    simdTilingData_->maxElement = ubAvailable;
    simdTilingData_->indiceUbSize = indexedDimNum_ * INDICES_SIZE;
    simdTilingData_->inputDtypeSize = inputDtypeSize_;
    simdTilingData_->indexedDimNum = indexedDimNum_;
    for (int i = 0; i < NUM_EIGHT; i++) {
        simdTilingData_->mergeInputShape[i] = mergeInputShape_[i];
        simdTilingData_->mergeInputIndexed[i] = mergeInputIndexed_[i];
        simdTilingData_->mergeOutputShape[i] = mergeOutputShape_[i];
        simdTilingData_->mergeOutToInput[i] = mergeOutToInput_[i];
        simdTilingData_->indicesToInput[i] = indicesToInput_[i];
    }
    simdTilingData_->mergeInputShapeDim = mergeInputShapeDim_;
    simdTilingData_->mergeOutputShapeDim = mergeOutputShapeDim_;
    simdTilingData_->isZeroOneZero = isZeroOneZero_;
    simdTilingData_->indexSize = indexSize_;
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
 * 不连续，则：out:     [gather_size, ..., other_size, ..., inner_size]
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

uint64_t IndexTilingSimd::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(0, 0, 0, 1, 0, 0, 0);
}

ge::graphStatus IndexTilingSimd::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Index, IndexTilingSimd, 20);
} // namespace optiling
