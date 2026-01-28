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
 * \brief index_tiling_full_load.cpp
 */

#include <vector>
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"
#include "util/math_util.h"
#include "index_tiling.h"
#include "index_tiling_full_load.h"

namespace optiling {
static constexpr size_t X_INPUT_IDX = 0;
static constexpr size_t MASK_INPUT_IDX = 1;
static constexpr size_t INDICES_IDX = 3;
static constexpr size_t Y_OUTPUT_IDX = 0;
static constexpr size_t DIM_THRESHOLD = 2;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t INDEX_THRESHOLD = 16;
static constexpr int64_t LAST_AXIS_THRESHOLD = 64;
static constexpr uint64_t TILING_KEY_FULL_LOAD_BASE = 200;
static constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t B8 = 1;
static constexpr int64_t B16 = 2;
static constexpr int64_t B32 = 4;
static constexpr int64_t B64 = 8;
static constexpr int64_t MAX_INT32_NUM = 2147483647;
static constexpr int64_t MAX_UINT16_NUM = 65535;
static constexpr int64_t TILING_KEY_MIDDLE_RATE = 10;
static constexpr uint8_t MASK_MODE_0 = 0;
static constexpr uint8_t MASK_MODE_1 = 1;
static constexpr uint8_t MASK_MODE_INVALID = 127;

static const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64,
                                                     ge::DT_BOOL,  ge::DT_BF16,    ge::DT_INT8,  ge::DT_UINT8};
inline static bool IsSupportDtype(const std::set<ge::DataType>& supportDtype, const ge::DataType dtype)
{
    return (supportDtype.count(dtype) != 0);
}
ge::graphStatus IndexFullLoadTiling::GetShapeAttrsInfo()
{
    auto xInputShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInputShape);
    inputShape_ = Ops::Base::EnsureNotScalar(xInputShape->GetStorageShape());

    auto maskInputShape = context_->GetInputShape(MASK_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskInputShape);
    maskShape_ = Ops::Base::EnsureNotScalar(maskInputShape->GetStorageShape());

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
    int64_t inputDataTypeSize = ge::GetSizeByDataType(inputDtype_);
    int64_t inputShapeSize = inputShape_.GetShapeSize();
    if (maskIndices_ != indicesTensorNum_) {
        return false;
    }
    if ((inputDataTypeSize == B8 || inputDataTypeSize == B16) && inputShapeSize > MAX_UINT16_NUM) {
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
    if (indicesDtypeSize == B64) {
        ubIndices_ =
            (static_cast<int64_t>(ubSize_) - inputSize - DOUBLE * blockSize_ - indicesTensorNum_ * blockSize_ * DOUBLE -
             blockSize_) /
            (lastAxisSize_ * inputOutputDtypeSize * DOUBLE + B32 * indicesTensorNum_ * DOUBLE + indicesDtypeSize);
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

ge::graphStatus IndexFullLoadTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const IndexCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->core_num;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
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
    tilingData_.set_eachCoreSize(eachCoreSize);
    tilingData_.set_normalCoreLoop(normalCoreLoop);
    tilingData_.set_normalCoreTail(normalCoreTail);
    tilingData_.set_tailCoreLoop(tailCoreLoop);
    tilingData_.set_tailCoreTail(tailCoreTail);
    tilingData_.set_indicesNum(indicesTensorNum_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_ubIndices(ubIndices_);
    tilingData_.set_lastDimSize(lastAxisSize_);
    tilingData_.set_inputShapeSize(inputShape_.GetShapeSize());
    std::vector<uint32_t> inputShapeVector(MAX_DIM_NUM, 0);
    int64_t baseOffset = MAX_DIM_NUM - inputShape_.GetDimNum();
    for (size_t i = 0; i < inputShape_.GetDimNum(); i++) {
        inputShapeVector[baseOffset + i] = inputShape_.GetDim(i);
    }
    uint32_t tmpArray[MAX_DIM_NUM] = {0};
    std::copy(inputShapeVector.begin(), inputShapeVector.end(), tmpArray);
    tilingData_.set_inputShape(tmpArray);
    uint32_t tmpArray1[MAX_DIM_NUM] = {0};
    std::vector<uint32_t> inputStrideVector(MAX_DIM_NUM, 0);
    uint32_t stride = 1;
    size_t inputDimNum = inputShape_.GetDimNum();
    for (int64_t i = inputDimNum - 1; i >= 0; --i) {
        tmpArray1[i] = stride;
        stride *= inputShape_.GetDim(i);
    }
    for (size_t i = 0; i < inputDimNum; i++) {
        inputStrideVector[MAX_DIM_NUM - inputDimNum + i] = tmpArray1[i];
    }
    std::copy(inputStrideVector.begin(), inputStrideVector.end(), tmpArray);
    tilingData_.set_inputStride(tmpArray);
    int64_t inputOutputDtypeSize = ge::GetSizeByDataType(inputDtype_);
    int64_t inputQueueSize = Ops::Base::CeilAlign(inputShape_.GetShapeSize() * inputOutputDtypeSize, blockSize_);
    int64_t indicesDtypeSize = ge::GetSizeByDataType(indicesDtype_);
    int64_t indicesQueueSize = 0;
    if (indicesDtypeSize == B64) {
        indicesQueueSize = Ops::Base::CeilAlign(ubIndices_ * B32, blockSize_) * indicesTensorNum_;
    } else {
        indicesQueueSize = Ops::Base::CeilAlign(ubIndices_ * indicesDtypeSize, blockSize_) * indicesTensorNum_;
    }
    int64_t outputQueueSize = Ops::Base::CeilAlign(ubIndices_ * lastAxisSize_ * inputOutputDtypeSize, blockSize_);
    tilingData_.set_inputQueSize(inputQueueSize);
    tilingData_.set_indicesQueSize(indicesQueueSize);
    tilingData_.set_outputQueSize(outputQueueSize);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IndexFullLoadTiling::DoLibApiTiling()
{
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
    // base + dim_num * 10 + maskMode
    return TILING_KEY_FULL_LOAD_BASE + inputShape_.GetDimNum() * TILING_KEY_MIDDLE_RATE + maskMode_;
}

ge::graphStatus IndexFullLoadTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IndexFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(Index, IndexFullLoadTiling, 10);
} // namespace optiling
