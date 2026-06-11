/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file foreach_regbase_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "log/error_code.h"
#include "foreach_regbase_tiling.h"

namespace optiling {
static constexpr uint64_t WORK_SPACE_SIZE = 32;
static constexpr uint64_t TPL_REGISTER_PRIORITY = 30000;
static constexpr uint64_t SIZE_2 = 2;

const char* ForeachRegbaseTiling::GetFirstTensorName() const
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    if (computeNodeInfo != nullptr && computeNodeInfo->GetIrInputsNum() > SIZE_2) {
        return "x1";
    }
    return "x";
}
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t TILING_KEY_HALF = 10001;
static constexpr uint64_t TILING_KEY_FLOAT = 10002;
static constexpr uint64_t TILING_KEY_INT = 10003;
static constexpr uint64_t TILING_KEY_BF16 = 10004;
static constexpr int64_t SINGLE_CORE_PROCESS_DATA = 4096;
static constexpr int64_t FIRST_INPUT_IDX = 0;
static constexpr int64_t SECOND_INPUT_IDX = 1;
static constexpr int64_t THIRD_INPUT_IDX = 2;
static constexpr int64_t THREE_INPUTS = 3;
static constexpr int64_t TWO_INPUTS = 2;
static constexpr int32_t MAX_SUPPORT_DIM_NUMS = 8;

static const std::vector<std::vector<ge::DataType>> SUPPORT_DTYPE_COMB = {
    {ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_FLOAT},
    {ge::DT_INT32, ge::DT_INT32}};
static const std::vector<std::vector<ge::DataType>> SCALAR_LIST_SUPPORT_DTYPE_COMB = {
    {ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT16, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_FLOAT},
    {ge::DT_INT32, ge::DT_INT64}};
ge::graphStatus ForeachRegbaseTiling::GetShapeAttrsInfo()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr, OP_LOGE(context_, "GetComputeNodeInfo failed."), return ge::GRAPH_FAILED);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(FIRST_INPUT_IDX);
    OP_CHECK_IF(
        anchorInstanceInfo == nullptr, OP_LOGE(context_, "GetInputInstanceInfo failed."), return ge::GRAPH_FAILED);
    totalTensorCount_ = anchorInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(
        totalTensorCount_ > MAX_TENSOR_CONT_950 || totalTensorCount_ <= 0,
        OP_LOGE_FOR_INVALID_TENSORNUM(
            context_->GetNodeName(), GetFirstTensorName(),
            static_cast<int64_t>(totalTensorCount_),
            ("within the range [1, " + std::to_string(MAX_TENSOR_CONT_950) + "]").c_str()),
        return ge::GRAPH_FAILED);
    totalDataCount_ = 0;
    dataType_ = ge::DT_UNDEFINED;
    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto tempDesc = context_->GetDynamicInputDesc(0, i);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(context_, "The input %u desc is null.", i), return ge::GRAPH_FAILED);
        auto srcDtype = tempDesc->GetDataType();
        // Determine whether all data types are consistent.
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = srcDtype;
        } else if (srcDtype != dataType_) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), GetFirstTensorName(),
                ge::TypeUtils::DataTypeToSerialString(srcDtype).c_str(),
                ("The dtypes of all tensors in the tensor list must be the same. "
                 "Currently, the dtype of the " + std::to_string(i) + "th tensor is inconsistent with that (" +
                 ge::TypeUtils::DataTypeToSerialString(dataType_) + ") of other tensors").c_str());
            return ge::GRAPH_FAILED;
        }
        auto tempShape = context_->GetDynamicInputShape(0, i);
        OP_CHECK_IF(tempShape == nullptr, OP_LOGE(context_, "The input %u shape is null.", i), return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(
            tempShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(),
                GetFirstTensorName(),
                std::to_string(tempShape->GetStorageShape().GetDimNum()).c_str(),
                "The shape dim of the " + std::to_string(i) + "th tensor in the tensor list should be less than or equal to 8"),
            return ge::GRAPH_FAILED);

        // Make a 32-byte alignment for each Tensor
        tensorDataCountList_[i] = tempShape->GetStorageShape().GetShapeSize();
        if (CheckShapeAllPositive(tempShape->GetStorageShape(), i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        totalDataCount_ += tensorDataCountList_[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTiling::CheckScalar(int64_t scalarIdx)
{
    auto scalarDesc = context_->GetRequiredInputDesc(scalarIdx);
    OP_CHECK_IF(scalarDesc == nullptr, OP_LOGE(context_, "The scalar desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc->GetDataType();
    std::vector<ge::DataType> dtypeComb = {dataType_, scalarDtype_};
    OP_CHECK_IF(
        std::find(SUPPORT_DTYPE_COMB.begin(), SUPPORT_DTYPE_COMB.end(), dtypeComb) == SUPPORT_DTYPE_COMB.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "scalar",
            (ge::TypeUtils::DataTypeToSerialString(dataType_) + " and " +
             ge::TypeUtils::DataTypeToSerialString(scalarDtype_)).c_str(),
            "The dtypes of x and scalar must be within the range {F32/F32, INT32/INT32, BF16/F32, F16/F16, F16/F32}"),
        return ge::GRAPH_FAILED);
    auto scalarShape = context_->GetRequiredInputShape(scalarIdx);
    OP_CHECK_IF(scalarShape == nullptr, OP_LOGE(context_, "The scalar shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "scalar",
            std::to_string(scalarShape->GetStorageShape().GetDimNum()).c_str(),
            "less than or equal to 8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetShapeSize() != 1,
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "scalar",
            std::to_string(scalarShape->GetStorageShape().GetShapeSize()).c_str(), "1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTiling::CheckScalarList(int64_t scalarIdx)
{
    auto scalarDesc = context_->GetRequiredInputDesc(scalarIdx);
    OP_CHECK_IF(scalarDesc == nullptr, OP_LOGE(context_, "The scalars desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc->GetDataType();
    OP_CHECK_IF(
        scalarDtype_ != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "scalars",
            ge::TypeUtils::DataTypeToSerialString(scalarDtype_).c_str(), "FP32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dataType_ != ge::DT_FLOAT && dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(dataType_).c_str(), "FP32, FP16 or BF16"),
        return ge::GRAPH_FAILED);
    auto scalarShape = context_->GetRequiredInputShape(scalarIdx);
    OP_CHECK_IF(scalarShape == nullptr, OP_LOGE(context_, "The scalars shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetShapeSize() != totalTensorCount_,
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "scalars",
            std::to_string(scalarShape->GetStorageShape().GetShapeSize()).c_str(),
            (std::to_string(totalTensorCount_)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTiling::CheckScalarListInt(int64_t scalarIdx)
{
    OP_CHECK_IF(
        dataType_ != ge::DT_FLOAT && dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 &&
            dataType_ != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(dataType_).c_str(), "FP32, FP16, BF16 or INT32"),
        return ge::GRAPH_FAILED);
    auto scalarDesc_1 = context_->GetRequiredInputDesc(scalarIdx);
    OP_CHECK_IF(scalarDesc_1 == nullptr, OP_LOGE(context_, "The scalars desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc_1->GetDataType();
    std::vector<ge::DataType> dtypeComb = {dataType_, scalarDtype_};
    OP_CHECK_IF(
        std::find(SCALAR_LIST_SUPPORT_DTYPE_COMB.begin(), SCALAR_LIST_SUPPORT_DTYPE_COMB.end(), dtypeComb) ==
            SCALAR_LIST_SUPPORT_DTYPE_COMB.end(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and scalars",
            (ge::TypeUtils::DataTypeToSerialString(dataType_) + " and " +
             ge::TypeUtils::DataTypeToSerialString(scalarDtype_)).c_str(),
            "The dtypes of x and scalars must be within the range {F32/F32, INT32/INT64, BF16/F32, F16/F32}"),
        return ge::GRAPH_FAILED);

    auto scalarShape_1 = context_->GetRequiredInputShape(scalarIdx);
    OP_CHECK_IF(scalarShape_1 == nullptr, OP_LOGE(context_, "The scalars shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape_1->GetStorageShape().GetShapeSize() != totalTensorCount_,
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "scalars",
            std::to_string(scalarShape_1->GetStorageShape().GetShapeSize()).c_str(),
            (std::to_string(totalTensorCount_)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTiling::DoOpTiling()
{
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    int64_t elementsPerBlock = SINGLE_CORE_PROCESS_DATA / sizePerElem;

    numBlocks_ = std::min<uint64_t>(aicoreParams_.numBlocks, MAX_CORE_CONT_950);
    uint32_t tempCoreNum = Ops::Base::CeilDiv(totalDataCount_, elementsPerBlock);
    if (tempCoreNum < numBlocks_) {
        numBlocks_ = tempCoreNum;
    }
    if (totalDataCount_ == 0) {
        numBlocks_ = 1;
        tensorStartList_[0] = 0;
        tensorEndList_[0] = 0;
        tensorStartOffsetList_[0] = 0;
        tensorEndOffsetList_[0] = -1;
    } else {
        AssignDataToEachCore(numBlocks_, elementsPerBlock);
    }

    foreachSoloTilingData_.set_tensorDataCountList(tensorDataCountList_);
    foreachSoloTilingData_.set_tensorStartList(tensorStartList_);
    foreachSoloTilingData_.set_tensorEndList(tensorEndList_);
    foreachSoloTilingData_.set_tensorStartOffsetList(tensorStartOffsetList_);
    foreachSoloTilingData_.set_tensorEndOffsetList(tensorEndOffsetList_);
    return ge::GRAPH_SUCCESS;
}

void ForeachRegbaseTiling::AssignDataToEachCore(int64_t needCoreNum, int64_t elementsPerBlock)
{
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = Ops::Base::CeilDiv(totalDataCount_, elementsPerBlock);
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList_[coreIndex] = 0;
    tensorStartOffsetList_[coreIndex] = 0;
    for (uint16_t j = 0; j < totalTensorCount_; j++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountList_[j] - cursorPos;

        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList_[coreIndex] = j;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList_[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountList_[j]) {
            tensorStartList_[coreIndex] = j;
            tensorStartOffsetList_[coreIndex] = cursorPos;
            --j; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList_[coreIndex] = j + 1;
            tensorStartOffsetList_[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount > 0) {
        tensorEndList_[coreIndex] = totalTensorCount_ - 1;
        tensorEndOffsetList_[coreIndex] = tensorDataCountList_[totalTensorCount_ - 1] - 1;
    }
}

ge::graphStatus ForeachRegbaseTiling::PostTiling()
{
    context_->SetBlockDim(numBlocks_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = WORK_SPACE_SIZE;
    auto tilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    foreachSoloTilingData_.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(foreachSoloTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTiling::CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(),
                GetFirstTensorName(),
                std::to_string(shape.GetDim(i)).c_str(),
                ("The " + std::to_string(i) + "th axis of the " + std::to_string(idx) + "th tensor in the tensor list must be 0 or a positive number").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachRegbaseTiling::GetTilingKey() const
{
    switch (dataType_) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_INT32:
            return TILING_KEY_INT;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return 0;
    }
}

ge::graphStatus ForeachRegbaseTiling::CheckOutput()
{
    size_t outputCount = context_->GetComputeNodeOutputNum();
    // In-place foreach ops declare no output (x serves as both input and output); skip the
    // output validation for them. Non-in-place ops keep the original {x, y} count/dtype/shape checks.
    if (outputCount == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        totalTensorCount_ != outputCount,
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context_->GetNodeName(), "x and y",
            (std::to_string(totalTensorCount_) + " and " + std::to_string(outputCount)).c_str(),
            "The tensor nums in {x, y} must be the same"),
        return ge::GRAPH_FAILED);
    for (uint32_t j = 0; j < totalTensorCount_; j++) {
        auto tempDesc = context_->GetOutputDesc(j);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(context_, "The output %u desc is null.", j), return ge::GRAPH_FAILED);
        auto dstDtype = tempDesc->GetDataType();
        OP_CHECK_IF(
            dstDtype != dataType_,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "y",
                ge::TypeUtils::DataTypeToSerialString(dstDtype).c_str(),
                "The dtype of y must be the same as x"),
            return ge::GRAPH_FAILED);
        auto srcShape = context_->GetDynamicInputShape(0, j);
        OP_CHECK_IF(srcShape == nullptr, OP_LOGE(context_, "The input %u shape is null.", j), return ge::GRAPH_FAILED);
        auto dstShape = context_->GetOutputShape(j);
        OP_CHECK_IF(dstShape == nullptr, OP_LOGE(context_, "The output %u shape is null.", j), return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(
            dstShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "y",
                std::to_string(dstShape->GetStorageShape().GetDimNum()).c_str(),
                ("The " + std::to_string(j) + "th tensor in tensor list y must be less than or equal to 8").c_str()),
            return ge::GRAPH_FAILED);

        if (srcShape->GetStorageShape() != dstShape->GetStorageShape() &&
                srcShape->GetStorageShape().GetShapeSize() > dstShape->GetStorageShape().GetShapeSize()) {
            std::string reasonMsg = "The shape size of " + std::to_string(j) +
                                    "th tensor in tensor list y should be greater than or equal to that of the tensor "
                                    "in the same position of the another tensor list x";
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context_->GetNodeName(), "y",
                (std::to_string(srcShape->GetStorageShape().GetShapeSize()) + " and " +
                 std::to_string(dstShape->GetStorageShape().GetShapeSize()))
                    .c_str(),
                reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalar::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalar(SECOND_INPUT_IDX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalar::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER + DOUBLE_BUFFER * sizePerElem;
    if (dataType_ == ge::DT_BF16 || dataType_ == ge::DT_FLOAT16) {
        ubSizePerNumber += sizeof(float);
    }
    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = ubSizeCanUse / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnary::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarList::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalarList(SECOND_INPUT_IDX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarList::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER + sizePerElem * DOUBLE_BUFFER;
    if (dataType_ == ge::DT_BF16 || dataType_ == ge::DT_FLOAT16) {
        ubSizePerNumber += sizeof(float);
    }
    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = ubSizeCanUse / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarListWithInt::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalarListInt(SECOND_INPUT_IDX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingTernaryScalar::CheckContext()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr, OP_LOGE(context_, "GetComputeNodeInfo failed."), return ge::GRAPH_FAILED);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(FIRST_INPUT_IDX);
    totalTensorCount_ = anchorInstanceInfo->GetInstanceNum();

    auto anchorInstanceInfoSecond = computeNodeInfo->GetInputInstanceInfo(SECOND_INPUT_IDX);
    OP_CHECK_IF(
        anchorInstanceInfoSecond == nullptr, OP_LOGE(context_, "GetInputInstanceInfo failed."),
        return ge::GRAPH_FAILED);
    uint16_t totalTensorCountSecond = anchorInstanceInfoSecond->GetInstanceNum();

    auto anchorInstanceInfoThird = computeNodeInfo->GetInputInstanceInfo(THIRD_INPUT_IDX);
    OP_CHECK_IF(
        anchorInstanceInfoThird == nullptr, OP_LOGE(context_, "GetInputInstanceInfo failed."), return ge::GRAPH_FAILED);
    uint16_t totalTensorCountThird = anchorInstanceInfoThird->GetInstanceNum();

    OP_CHECK_IF(
        totalTensorCount_ != totalTensorCountSecond || totalTensorCount_ != totalTensorCountThird,
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and x3",
            (std::to_string(totalTensorCount_) + ", " + std::to_string(totalTensorCountSecond) + " and " +
             std::to_string(totalTensorCountThird)).c_str(),
            "The tensor nums in {x1, x2, x3} must be the same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingTernaryScalar::CheckShape(uint32_t idx)
{
    auto tempShapeFirst = context_->GetDynamicInputShape(FIRST_INPUT_IDX, idx);
    OP_CHECK_IF(
        tempShapeFirst == nullptr, OP_LOGE(context_, "The input1 %u shape is null.", idx), return ge::GRAPH_FAILED);
    auto tempShapeSecond = context_->GetDynamicInputShape(SECOND_INPUT_IDX, idx);
    OP_CHECK_IF(
        tempShapeSecond == nullptr, OP_LOGE(context_, "The input2 %u shape is null.", idx), return ge::GRAPH_FAILED);
    auto tempShapeThird = context_->GetDynamicInputShape(THIRD_INPUT_IDX, idx);
    OP_CHECK_IF(
        tempShapeThird == nullptr, OP_LOGE(context_, "The input3 %u shape is null.", idx), return ge::GRAPH_FAILED);
    // check max dim
    OP_CHECK_IF(
        tempShapeSecond->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "x2",
            std::to_string(tempShapeSecond->GetStorageShape().GetDimNum()).c_str(),
            ("The shape dim of " + std::to_string(idx) + "th tensor in the tensor list x2 must be less than or equal to 8").c_str()),
        return ge::GRAPH_FAILED);
    // check max dim
    OP_CHECK_IF(
        tempShapeThird->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x3",
            std::to_string(tempShapeThird->GetStorageShape().GetDimNum()).c_str(),
            ("The shape dim of " + std::to_string(idx) + "th tensor in the tensor list x3 must be less than or equal to 8").c_str()),
        return ge::GRAPH_FAILED);
    if (tempShapeFirst->GetStorageShape().GetShapeSize() != tempShapeSecond->GetStorageShape().GetShapeSize() ||
            tempShapeFirst->GetStorageShape().GetShapeSize() != tempShapeThird->GetStorageShape().GetShapeSize()) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and x3",
            (std::to_string(tempShapeFirst->GetStorageShape().GetShapeSize()) + ", " +
             std::to_string(tempShapeSecond->GetStorageShape().GetShapeSize()) + " and " +
             std::to_string(tempShapeThird->GetStorageShape().GetShapeSize())).c_str(),
            "The shape sizes of x1, x2 and x3 must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingTernaryScalar::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto tempDescSecond = context_->GetDynamicInputDesc(SECOND_INPUT_IDX, i);
        OP_CHECK_IF(
            tempDescSecond == nullptr, OP_LOGE(context_, "The input2 %u desc is null.", i), return ge::GRAPH_FAILED);
        auto tempDescThird = context_->GetDynamicInputDesc(THIRD_INPUT_IDX, i);
        OP_CHECK_IF(
            tempDescThird == nullptr, OP_LOGE(context_, "The input3 %u desc is null.", i), return ge::GRAPH_FAILED);
        // check datatype
        auto srcDtypeSecond = tempDescSecond->GetDataType();
        auto srcDtypeThird = tempDescThird->GetDataType();
        OP_CHECK_IF(
            dataType_ != srcDtypeSecond || dataType_ != srcDtypeThird,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x1, x2 and x3",
                (ge::TypeUtils::DataTypeToSerialString(dataType_) + ", " +
                 ge::TypeUtils::DataTypeToSerialString(srcDtypeSecond) + " and " +
                 ge::TypeUtils::DataTypeToSerialString(srcDtypeThird)).c_str(),
                "The dtypes of x1, x2 and x3 must be the same"), return ge::GRAPH_FAILED);

        if (CheckShape(i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalar(THREE_INPUTS) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingTernaryScalar::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);

    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);

    // 3个输入加1个输出
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER * THREE_INPUTS + sizePerElem * DOUBLE_BUFFER;

    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = ubSizeCanUse / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinaryScalar::CheckContext()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr, OP_LOGE(context_, "GetComputeNodeInfo failed."), return ge::GRAPH_FAILED);
    auto anchorInstanceInfoSecond = computeNodeInfo->GetInputInstanceInfo(SECOND_INPUT_IDX);
    OP_CHECK_IF(
        anchorInstanceInfoSecond == nullptr, OP_LOGE(context_, "GetInputInstanceInfo failed."),
        return ge::GRAPH_FAILED);
    uint16_t totalTensorCountSecond = anchorInstanceInfoSecond->GetInstanceNum();

    OP_CHECK_IF(
        totalTensorCount_ != totalTensorCountSecond,
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context_->GetNodeName(), "x1 and x2",
            (std::to_string(totalTensorCount_) + " and " + std::to_string(totalTensorCountSecond)).c_str(),
            "The tensor nums in {x1, x2} must be the same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinaryScalar::CheckShape(uint32_t idx)
{
    auto tempShapeFirst = context_->GetDynamicInputShape(FIRST_INPUT_IDX, idx);
    OP_CHECK_IF(
        tempShapeFirst == nullptr, OP_LOGE(context_, "The input1 %u shape is null.", idx), return ge::GRAPH_FAILED);
    auto tempShapeSecond = context_->GetDynamicInputShape(SECOND_INPUT_IDX, idx);
    OP_CHECK_IF(
        tempShapeSecond == nullptr, OP_LOGE(context_, "The input2 %u shape is null.", idx), return ge::GRAPH_FAILED);
    // check max dim
    OP_CHECK_IF(
        tempShapeSecond->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x2",
            std::to_string(tempShapeSecond->GetStorageShape().GetDimNum()).c_str(),
            ("The shape dim of the " + std::to_string(idx) + "th tensor of x2 must be less than or equal to 8").c_str()),
        return ge::GRAPH_FAILED);
    if (tempShapeFirst->GetStorageShape().GetShapeSize() != tempShapeSecond->GetStorageShape().GetShapeSize()) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2",
            (std::to_string(tempShapeFirst->GetStorageShape().GetShapeSize()) + " and " +
             std::to_string(tempShapeSecond->GetStorageShape().GetShapeSize())).c_str(),
            "The shape sizes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinaryScalar::CheckScalar()
{
    auto scalarDesc = context_->GetRequiredInputDesc(THIRD_INPUT_IDX);
    OP_CHECK_IF(scalarDesc == nullptr, OP_LOGE(context_, "The scalar desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc->GetDataType();
    OP_CHECK_IF(
        scalarDtype_ != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "weight",
            ge::TypeUtils::DataTypeToSerialString(scalarDtype_).c_str(), "FP32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dataType_ != ge::DT_FLOAT && dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1",
            ge::TypeUtils::DataTypeToSerialString(dataType_).c_str(), "FP32, FP16 or BF16"),
        return ge::GRAPH_FAILED);
    auto scalarShape = context_->GetRequiredInputShape(THIRD_INPUT_IDX);
    OP_CHECK_IF(scalarShape == nullptr, OP_LOGE(context_, "The scalar shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "weight",
            std::to_string(scalarShape->GetStorageShape().GetDimNum()).c_str(),
            "less than or equal to 8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetShapeSize() != 1,
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "weight",
            std::to_string(scalarShape->GetStorageShape().GetShapeSize()).c_str(), "1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinaryScalar::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto tempDescSecond = context_->GetDynamicInputDesc(SECOND_INPUT_IDX, i);
        OP_CHECK_IF(
            tempDescSecond == nullptr, OP_LOGE(context_, "The input2 %u desc is null.", i), return ge::GRAPH_FAILED);
        // check datatype
        auto srcDtypeSecond = tempDescSecond->GetDataType();
        OP_CHECK_IF(
            dataType_ != srcDtypeSecond,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x1 and x2",
                (ge::TypeUtils::DataTypeToSerialString(dataType_) + " and " +
                 ge::TypeUtils::DataTypeToSerialString(srcDtypeSecond)).c_str(),
                "The dtypes of x1 and x2 must be the same"),
            return ge::GRAPH_FAILED);

        if (CheckShape(i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalar() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinaryScalar::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);

    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);

    // 2个输入加1个输出
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER * TWO_INPUTS + sizePerElem * DOUBLE_BUFFER;

    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = ubSizeCanUse / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachRegbaseTilingBinaryScalar::GetTilingKey() const
{
    switch (dataType_) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return 0;
    }
}

ge::graphStatus ForeachRegbaseTilingBinary::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    // The second tensor list (x2) must match x1's dtype.
    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto descSecond = context_->GetDynamicInputDesc(SECOND_INPUT_IDX, i);
        OP_CHECK_IF(
            descSecond == nullptr, OP_LOGE(context_, "The input2 %u desc is null.", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            descSecond->GetDataType() != dataType_,
            OP_LOGE(context_, "The dtypes of x1 and x2 must be the same."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// Shared UB-split tail: convert per-element UB bytes into inputsTensorUbSize and store it.
// Self-guards its divisors (ubSizePerNumber, sizePerElem) so the divisions stay divide-by-zero safe;
// blockSize from GetUbBlockSize is a fixed 32 and FloorAlign itself returns 0 when the align is 0.
// Currently used by ForeachRegbaseTilingBinary; the pre-existing UnaryScalar/UnaryScalarList/
// UnaryScalarList2 paths keep their inline copies and can adopt this helper in a later pass.
ge::graphStatus ForeachRegbaseTiling::SetInputsTensorUbSize(int64_t ubSizePerNumber)
{
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        ubSizePerNumber <= 0 || sizePerElem <= 0,
        OP_LOGE(context_, "Invalid UB divisor: ubSizePerNumber=%ld, sizePerElem=%ld.", ubSizePerNumber, sizePerElem),
        return ge::GRAPH_FAILED);
    int64_t inputsTensorUbSize = aicoreParams_.ubSize / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingBinary::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    // 2 inputs + 1 output, double buffered
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER * TWO_INPUTS + sizePerElem * DOUBLE_BUFFER;
    // half/bfloat16 binary ops that compute in float (e.g. div) need two float cast buffers.
    if (dataType_ == ge::DT_BF16 || dataType_ == ge::DT_FLOAT16) {
        ubSizePerNumber += static_cast<int64_t>(sizeof(float)) * TWO_INPUTS;
    }
    OP_CHECK_IF(
        SetInputsTensorUbSize(ubSizePerNumber) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "Set inputsTensorUbSize failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarList2::CheckScalarList(int64_t scalarIdx)
{
    auto scalarDesc = context_->GetRequiredInputDesc(scalarIdx);
    OP_CHECK_IF(scalarDesc == nullptr, OP_LOGE(context_, "The scalars desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc->GetDataType();
    std::vector<ge::DataType> dtypeComb = {dataType_, scalarDtype_};
    OP_CHECK_IF(
        std::find(SCALAR_LIST_SUPPORT_DTYPE_COMB.begin(), SCALAR_LIST_SUPPORT_DTYPE_COMB.end(), dtypeComb) ==
            SCALAR_LIST_SUPPORT_DTYPE_COMB.end(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x and scalars",
            (ge::TypeUtils::DataTypeToSerialString(dataType_) + " and " +
             ge::TypeUtils::DataTypeToSerialString(scalarDtype_)).c_str(),
            "The dtypes of x and scalars must be within the range {F32/F32, INT32/INT64, BF16/F32, F16/F32}"),
        return ge::GRAPH_FAILED);
    auto scalarShape = context_->GetRequiredInputShape(scalarIdx);
    OP_CHECK_IF(scalarShape == nullptr, OP_LOGE(context_, "The scalars shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetShapeSize() != totalTensorCount_,
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "scalars",
            std::to_string(scalarShape->GetStorageShape().GetShapeSize()).c_str(),
            (std::to_string(totalTensorCount_)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarList2::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalarList(SECOND_INPUT_IDX) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachRegbaseTilingUnaryScalarList2::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Base tiling failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ForeachRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER + sizePerElem * DOUBLE_BUFFER;
    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = ubSizeCanUse / ubSizePerNumber;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ForeachAddScalar, ForeachRegbaseTilingUnaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachMulScalar, ForeachRegbaseTilingUnaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachDivScalar, ForeachRegbaseTilingUnaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachAddcmulScalar, ForeachRegbaseTilingTernaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachLerpScalar, ForeachRegbaseTilingBinaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachDivScalarList, ForeachRegbaseTilingUnaryScalarList, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachAddScalarList, ForeachRegbaseTilingUnaryScalarList2, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachMulScalarList, ForeachRegbaseTilingUnaryScalarList2, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachSqrt, ForeachRegbaseTilingUnary, TPL_REGISTER_PRIORITY);
// in-place foreach operators (Ascend 950)
REGISTER_OPS_TILING_TEMPLATE(ForeachMulScalarInplace, ForeachRegbaseTilingUnaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachSubScalarInplace, ForeachRegbaseTilingUnaryScalar, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachMulListInplace, ForeachRegbaseTilingBinary, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachDivListInplace, ForeachRegbaseTilingBinary, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachAddListInplace, ForeachRegbaseTilingBinary, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachSubListInplace, ForeachRegbaseTilingBinary, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachACosInplace, ForeachRegbaseTilingUnary, TPL_REGISTER_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(ForeachLogInplace, ForeachRegbaseTilingUnary, TPL_REGISTER_PRIORITY);
} // namespace optiling