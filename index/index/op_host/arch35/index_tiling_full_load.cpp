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
 * \file index_tiling_full_load.cpp
 * \brief Full load tiling implementation for Index operator
 */

#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "../../op_kernel/arch35/index_tiling_key.h"
#include "index_tiling.h"
#include "index_tiling_full_load.h"

using namespace Index;
namespace optiling {
static constexpr int64_t MAX_DIM_NUM = 8;
static constexpr size_t X_INPUT_IDX = 0;
static constexpr size_t MASK_INPUT_IDX = 1;
static constexpr size_t INDICES_IDX = 3;
static constexpr size_t Y_OUTPUT_IDX = 0;
static constexpr size_t DIM_THRESHOLD = 2;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t INDEX_THRESHOLD = 16;
static constexpr int64_t LAST_AXIS_THRESHOLD = 64;
static constexpr int64_t MAX_INT32_NUM = 2147483647;
static constexpr int64_t MAX_UINT16_NUM = 65535;
static constexpr uint8_t MASK_MODE_0 = 0;
static constexpr uint8_t MASK_MODE_1 = 1;
static constexpr uint8_t MASK_MODE_INVALID = 127;

ge::graphStatus IndexFullLoadTiling::GetShapeAttrsInfo()
{
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    // 获取 tiling 数据
    tilingData_ = context_->GetTilingData<IndexFullLoadTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_CHECK_IF(
        memset_s(tilingData_, sizeof(IndexFullLoadTilingData), 0, sizeof(IndexFullLoadTilingData)) != EOK,
        OP_LOGE(context_->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    auto xInputShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    inputShape_ = Ops::NN::OpTiling::EnsureNotScalar(xInputShape->GetStorageShape());

    auto maskInputShape = context_->GetInputShape(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskInputShape);
    maskShape_ = Ops::NN::OpTiling::EnsureNotScalar(maskInputShape->GetStorageShape());

    auto yOutputShape = context_->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOutputShape);
    outputShape_ = yOutputShape->GetStorageShape();

    auto xInput = context_->GetInputDesc(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInput);
    inputDtype_ = xInput->GetDataType();

    auto idxInput = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxInput);
    indicesDtype_ = idxInput->GetDataType();

    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, anchorInstanceInfo);
    indicesTensorNum_ = anchorInstanceInfo->GetInstanceNum();

    const gert::Tensor* maskTensor = context_->GetInputTensor(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskTensor);
    int64_t maskSize = static_cast<int64_t>(maskShape_.GetDim(0));
    for (int64_t i = 0; i < maskSize; i++) {
        if (maskTensor->GetData<int64_t>()[i] == 1) {
            maskIndices_++;
        }
    }

    return ge::GRAPH_SUCCESS;
}

bool IndexFullLoadTiling::IsCapable()
{
    if (inputShape_.GetDimNum() > DIM_THRESHOLD) {
        return false;
    }
    const int64_t dtypeSizeB8 = static_cast<int64_t>(DTYPE_SIZE_B8);
    const int64_t dtypeSizeB16 = static_cast<int64_t>(DTYPE_SIZE_B16);
    int64_t inputDataTypeSize = ge::GetSizeByDataType(inputDtype_);
    int64_t inputShapeSize = inputShape_.GetShapeSize();
    if (maskIndices_ != indicesTensorNum_) {
        return false;
    }
    if ((inputDataTypeSize == dtypeSizeB8 || inputDataTypeSize == dtypeSizeB16) &&
        inputShapeSize > MAX_UINT16_NUM) {
        return false;
    } else if (inputShapeSize > MAX_INT32_NUM) {
        return false;
    }
    if (SetMaskMode() != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (CalcUBBuffer() != ge::GRAPH_SUCCESS || ubIndices_ < INDEX_THRESHOLD) {
        return false;
    }
    if (lastAxisSize_ > LAST_AXIS_THRESHOLD) {
        return false;
    }
    return true;
}

// input all + MASK + INDICES TENSOR + output
// mask(1, 0) -> outputshape:indices.size * last axis
// mask(1, 1)/mask(1) -> outputshape:indices.size
ge::graphStatus IndexFullLoadTiling::CalcUBBuffer()
{
    int64_t dimSize = inputShape_.GetDimNum();
    int64_t inputOutputDtypeSize = ge::GetSizeByDataType(inputDtype_);
    int64_t indicesDtypeSize = ge::GetSizeByDataType(indicesDtype_);
    const int64_t dtypeSizeB32 = static_cast<int64_t>(DTYPE_SIZE_B32);
    const int64_t dtypeSizeB64 = static_cast<int64_t>(DTYPE_SIZE_B64);
    const gert::Tensor* maskTensor = context_->GetInputTensor(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskTensor);
    int64_t maskSize = static_cast<int64_t>(maskShape_.GetDim(0));
    for (int64_t i = dimSize - 1; i >= 0; i--) {
        if (i >= maskSize || maskTensor->GetData<int64_t>()[i] == 0) {
            lastAxisSize_ *= inputShape_.GetDim(i);
        } else {
            break;
        }
    }
    int64_t inputSize = Ops::Base::CeilAlign(inputShape_.GetShapeSize() * inputOutputDtypeSize, blockSize_);
    if (indicesDtypeSize == dtypeSizeB64) {
        ubIndices_ =
            (static_cast<int64_t>(ubSize_) - inputSize - DOUBLE * blockSize_ - indicesTensorNum_ * blockSize_ * DOUBLE -
             blockSize_) /
            (lastAxisSize_ * inputOutputDtypeSize * DOUBLE + dtypeSizeB32 * indicesTensorNum_ * DOUBLE +
             indicesDtypeSize);
    } else {
        ubIndices_ = (static_cast<int64_t>(ubSize_) - inputSize - DOUBLE * blockSize_ -
                      indicesTensorNum_ * blockSize_ * DOUBLE) /
                     (lastAxisSize_ * inputOutputDtypeSize * DOUBLE + indicesDtypeSize * indicesTensorNum_ * DOUBLE);
    }
    if (ubIndices_ <= 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFullLoadTiling::DoOpTiling()
{
    auto indicesTensorShapePtr = context_->GetDynamicInputShape(INDICES_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesTensorShapePtr);
    gert::Shape inputTensorShape = indicesTensorShapePtr->GetStorageShape();
    int64_t indicesSize = inputTensorShape.GetShapeSize();

    int64_t eachCoreSize = Ops::Base::CeilDiv(indicesSize, static_cast<int64_t>(coreNum_));
    eachCoreSize = std::max(eachCoreSize, INDEX_THRESHOLD);
    int64_t usedCoreNum = Ops::Base::CeilDiv(indicesSize, eachCoreSize);
    int64_t normalCoreLoop = Ops::Base::CeilDiv(eachCoreSize, ubIndices_);
    int64_t normalCoreTail = eachCoreSize - (normalCoreLoop - 1) * ubIndices_;
    int64_t tailCoreSize = indicesSize - (usedCoreNum - 1) * eachCoreSize;
    int64_t tailCoreLoop = Ops::Base::CeilDiv(tailCoreSize, ubIndices_);
    int64_t tailCoreTail = tailCoreSize - (tailCoreLoop - 1) * ubIndices_;
    usedCoreNum_ = usedCoreNum;

    // tilingData_ 已经在 GetShapeAttrsInfo 中初始化
    tilingData_->eachCoreSize = eachCoreSize;
    tilingData_->normalCoreLoop = normalCoreLoop;
    tilingData_->normalCoreTail = normalCoreTail;
    tilingData_->tailCoreLoop = tailCoreLoop;
    tilingData_->tailCoreTail = tailCoreTail;
    tilingData_->indicesNum = indicesTensorNum_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->ubIndices = ubIndices_;
    tilingData_->lastDimSize = lastAxisSize_;
    tilingData_->inputShapeSize = inputShape_.GetShapeSize();
    const size_t inputDimNum = inputShape_.GetDimNum();
    const size_t inputOffset = static_cast<size_t>(MAX_DIM_NUM) - inputDimNum;
    uint32_t stride = 1;
    for (int64_t i = static_cast<int64_t>(inputDimNum) - 1; i >= 0; --i) {
        const size_t tilingIdx = inputOffset + static_cast<size_t>(i);
        const uint32_t curDim = static_cast<uint32_t>(inputShape_.GetDim(i));
        tilingData_->inputShape[tilingIdx] = curDim;
        tilingData_->inputStride[tilingIdx] = stride;
        stride *= curDim;
    }
    int64_t inputOutputDtypeSize = ge::GetSizeByDataType(inputDtype_);
    int64_t inputQueueSize = Ops::Base::CeilAlign(inputShape_.GetShapeSize() * inputOutputDtypeSize, blockSize_);
    int64_t indicesDtypeSize = ge::GetSizeByDataType(indicesDtype_);
    const int64_t dtypeSizeB32 = static_cast<int64_t>(DTYPE_SIZE_B32);
    const int64_t dtypeSizeB64 = static_cast<int64_t>(DTYPE_SIZE_B64);
    int64_t indicesQueueSize = 0;
    if (indicesDtypeSize == dtypeSizeB64) {
        indicesQueueSize = Ops::Base::CeilAlign(ubIndices_ * dtypeSizeB32, blockSize_) * indicesTensorNum_;
    } else {
        indicesQueueSize = Ops::Base::CeilAlign(ubIndices_ * indicesDtypeSize, blockSize_) * indicesTensorNum_;
    }
    int64_t outputQueueSize = Ops::Base::CeilAlign(ubIndices_ * lastAxisSize_ * inputOutputDtypeSize, blockSize_);
    tilingData_->inputQueSize = inputQueueSize;
    tilingData_->indicesQueSize = indicesQueueSize;
    tilingData_->outputQueSize = outputQueueSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFullLoadTiling::SetMaskMode()
{
    const gert::Tensor* maskTensor = context_->GetInputTensor(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskTensor);
    int64_t dimSize = inputShape_.GetDimNum();
    int64_t maskValue[2] = {0, 0};
    int64_t maskSize = static_cast<int64_t>(maskShape_.GetDim(0));
    for (int i = 0; i < maskSize; i++) {
        maskValue[i] = maskTensor->GetData<int64_t>()[i];
    }
    if (dimSize == DIM_THRESHOLD) {
        if (maskValue[0] == 1 && maskValue[1] == 1) {
            maskMode_ = MASK_MODE_1;
        } else if (maskValue[0] == 1 && maskValue[1] == 0) {
            maskMode_ = MASK_MODE_0;
        } else {
            maskMode_ = MASK_MODE_INVALID;
        }
    } else if (dimSize == 1) {
        if (maskValue[0] == 1) {
            maskMode_ = MASK_MODE_1;
        } else {
            maskMode_ = MASK_MODE_INVALID;
        }
    } else {
        maskMode_ = MASK_MODE_INVALID;
    }
    if (maskMode_ == MASK_MODE_INVALID) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexFullLoadTiling::GetTilingKey() const
{
    uint32_t fullLoadMode = 0;
    if (inputShape_.GetDimNum() == 1) {
        fullLoadMode = INDEX_FULL_LOAD_1D;
    } else if (inputShape_.GetDimNum() == 2 && maskMode_ == MASK_MODE_0) {
        fullLoadMode = INDEX_FULL_LOAD_2D_MASK_0;
    } else if (inputShape_.GetDimNum() == 2 && maskMode_ == MASK_MODE_1) {
        fullLoadMode = INDEX_FULL_LOAD_2D_MASK_1;
    }
    return GET_TPL_TILING_KEY(GenXDtype(), fullLoadMode, 0, 0, 0, 0, 0);
}

ge::graphStatus IndexFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Index, IndexFullLoadTiling, 10);
} // namespace optiling
