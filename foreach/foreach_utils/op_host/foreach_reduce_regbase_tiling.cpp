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
 * \file foreach_reduce_regbase_tiling.cpp
 * \brief
 */
#include <algorithm>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "log/error_code.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_info_def.h"
#include "common_dtype.h"
#include "foreach_reduce_regbase_tiling.h"

namespace optiling {
static constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint64_t TILING_KEY_HALF_950 = 10001;
static constexpr uint64_t TILING_KEY_FLOAT_950 = 10002;
static constexpr uint64_t TILING_KEY_BF16_950 = 10004;
static constexpr int64_t SINGLE_CORE_PROCESS_DATA = 4096;
static constexpr int64_t FIRST_INPUT_IDX = 0;
static constexpr int64_t SECOND_INPUT_IDX = 1;
static constexpr int32_t MAX_SUPPORT_DIM_NUMS = 8;

ge::graphStatus ForeachReduceRegbaseTiling::GetShapeAttrsInfo()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr, OP_LOGE(context_, "GetComputeNodeInfo failed."), return ge::GRAPH_FAILED);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(FIRST_INPUT_IDX);
    OP_CHECK_IF(
        anchorInstanceInfo == nullptr, OP_LOGE(context_, "GetInputInstanceInfo failed."), return ge::GRAPH_FAILED);
    totalTensorCount_ = anchorInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(
        totalTensorCount_ > MAX_TENSOR_CONT_910D || totalTensorCount_ <= 0,
        OP_LOGE(
            context_, "The number of input tensors must not be greater than %hu, but get [%hu].", MAX_TENSOR_CONT_910D,
            totalTensorCount_),
        return ge::GRAPH_FAILED);
    totalDataCount_ = 0;
    dataType_ = ge::DT_UNDEFINED;
    maxTensorNumPerCore_ = 0;
    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto tempDesc = context_->GetDynamicInputDesc(0, i);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(context_, "The input %u desc is null.", i), return ge::GRAPH_FAILED);
        auto srcDtype = tempDesc->GetDataType();
        // Determine whether all data types are consistent.
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = srcDtype;
        } else if (srcDtype != dataType_) {
            OP_LOGE(context_, "DataType of all input should be same.");
            return ge::GRAPH_FAILED;
        }
        auto tempShape = context_->GetDynamicInputShape(0, i);
        OP_CHECK_IF(tempShape == nullptr, OP_LOGE(context_, "The input %u shape is null.", i), return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(
            tempShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
            OP_LOGE(
                context_->GetNodeName(),
                "The input1 tensors[%u] shape is invaild, and it cannot be larger than 8 dimensions, but its %zu dims.",
                i, tempShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);

        // Make a 32-byte alignment for each Tensor
        tensorDataCountList_[i] = tempShape->GetStorageShape().GetShapeSize();
        if (CheckShapeAllPositive(tempShape->GetStorageShape(), i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        totalDataCount_ += tensorDataCountList_[i];
    }

    // check scalar shape and dtype
    OP_CHECK_IF(
        CheckScalar() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling context check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceRegbaseTiling::CheckScalar()
{
    auto scalarDesc = context_->GetRequiredInputDesc(SECOND_INPUT_IDX);
    OP_CHECK_IF(scalarDesc == nullptr, OP_LOGE(context_, "The scalar desc is null."), return ge::GRAPH_FAILED);
    scalarDtype_ = scalarDesc->GetDataType();
    OP_CHECK_IF(
        scalarDtype_ != ge::DT_FLOAT && scalarDtype_ != ge::DT_INT64,
        OP_LOGE(
            context_, "The data type of the scalar only supports FP32 and INT64, but it is %s.",
            Ops::Base::ToString(scalarDtype_).c_str()),
        return ge::GRAPH_FAILED);
    auto scalarShape = context_->GetRequiredInputShape(SECOND_INPUT_IDX);
    OP_CHECK_IF(scalarShape == nullptr, OP_LOGE(context_, "The scalar shape is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
        OP_LOGE(
            context_, "The scalar shape is invaild, and it cannot be larger than 8 dimensions, but its %zu dims.",
            scalarShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        scalarShape->GetStorageShape().GetShapeSize() != 1, OP_LOGE(context_, "The scalar shape must be 1."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceRegbaseTiling::CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE(context_, "Dim %lu of input %u expect cant be negtive, but actual %ld.", i, idx, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceRegbaseTiling::AssignTensorMiddleCountList(int64_t needCoreNum)
{
    uint16_t preCoreTensorIndex = 0;
    for (uint32_t i = 1; i < needCoreNum; i++) {
        if (preCoreTensorIndex == tensorStartList_[i]) {
            tensorMiddleCountList_[preCoreTensorIndex] += 1;
        } else {
            if (tensorStartOffsetList_[i] > 0) {
                tensorMiddleCountList_[tensorStartList_[i]] += 1;
            }
        }
        preCoreTensorIndex = tensorStartList_[i];
    }
    uint16_t tensorMiddleStart = 0;
    for (uint32_t j = 0; j < totalTensorCount_; j++) {
        tensorMiddleCountList_[j]++;
        tensorMiddleStartList_[j] = tensorMiddleStart;
        tensorMiddleStart += tensorMiddleCountList_[j];
    }
    uint16_t coreMiddleOffset = 0;
    for (uint32_t j = 0; j < needCoreNum; j++) {
        coreMiddleOffsetList_[j] = coreMiddleOffset;
        coreMiddleOffset += tensorEndList_[j] - tensorStartList_[j] + 1;
    }
    return ge::GRAPH_SUCCESS;
}
void ForeachReduceRegbaseTiling::AssignDataToEachCore(int64_t needCoreNum, int64_t elementsPerBlock)
{
    // Kernel the input data according to 32 byte alignment.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t blockCount = Ops::Base::CeilDiv(totalDataCount_, elementsPerBlock);
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    uint64_t cursorPos = 0;
    tensorStartList_[coreIndex] = 0;
    tensorStartOffsetList_[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount_; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountList_[i] - cursorPos;

        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList_[coreIndex] = i;
        uint16_t tensorNumCurrentCore = (tensorEndList_[coreIndex] - tensorStartList_[coreIndex] + 1);
        maxTensorNumPerCore_ = std::max(maxTensorNumPerCore_, tensorNumCurrentCore);

        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList_[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountList_[i]) {
            tensorStartList_[coreIndex] = i;
            tensorStartOffsetList_[coreIndex] = cursorPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList_[coreIndex] = i + 1;
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
    // last core
    maxTensorNumPerCore_ = std::max(
        maxTensorNumPerCore_, static_cast<uint16_t>(tensorEndList_[coreIndex] - tensorStartList_[coreIndex] + 1));
}
ge::graphStatus ForeachReduceRegbaseTiling::CheckOutput()
{
    size_t outputCount = context_->GetComputeNodeOutputNum();
    OP_CHECK_IF(
        totalTensorCount_ != outputCount,
        OP_LOGE(
            context_, "The output num should be same with input, expect %hu, actual %lu.", totalTensorCount_,
            outputCount),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < totalTensorCount_; i++) {
        auto tempDesc = context_->GetOutputDesc(i);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(context_, "The output %u desc is null.", i), return ge::GRAPH_FAILED);
        auto dstDtype = tempDesc->GetDataType();
        OP_CHECK_IF(
            dstDtype != dataType_, OP_LOGE(context_, "The output %u datatype should be same with input.", i),
            return ge::GRAPH_FAILED);
        auto dstShape = context_->GetOutputShape(i);
        OP_CHECK_IF(dstShape == nullptr, OP_LOGE(context_, "The output %u shape is null.", i), return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(
            dstShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
            OP_LOGE(
                context_->GetNodeName(),
                "The output tensors[%u] shape is invaild, and it cannot be larger than 8 dimensions, but its %zu dims.",
                i, dstShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            dstShape->GetStorageShape().GetShapeSize() != 1,
            OP_LOGE(
                context_, "The output tensors[%u] shapeSize should be 1, but it is %ld.", i,
                dstShape->GetStorageShape().GetShapeSize()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceRegbaseTiling::DoOpTiling()
{
    OP_CHECK_IF(
        ForeachReduceRegbaseTiling::CheckOutput() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check output info failed."),
        return ge::GRAPH_FAILED);
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    OP_CHECK_IF(
        sizePerElem <= 0, OP_LOGE(context_, "The datatype size is neg: %ld.", sizePerElem), return ge::GRAPH_FAILED);
    int64_t elementsPerBlock = SINGLE_CORE_PROCESS_DATA / sizePerElem;

    numBlocks_ = std::min<uint64_t>(aicoreParams_.numBlocks, MAX_CORE_CONT_910D);
    uint32_t tempCoreNum = Ops::Base::CeilDiv(totalDataCount_, elementsPerBlock);
    if (tempCoreNum < numBlocks_) {
        numBlocks_ = tempCoreNum;
    }
    // Divisible, representing the amount of data each core needs to process.
    if (numBlocks_ == 0) {
        numBlocks_ = 1;
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

    // Reduce Op Addition
    AssignTensorMiddleCountList(numBlocks_);

    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);

    // out buffer 32位向上取整
    maxTensorNumPerCore_ =
        Ops::Base::FloorAlign(static_cast<int64_t>(maxTensorNumPerCore_ * sizeof(float) + blockSize - 1), blockSize) /
        sizeof(float);

    int64_t ubSizeCanUse = aicoreParams_.ubSize;
    int64_t inputsTensorUbSize = (ubSizeCanUse - maxTensorNumPerCore_ * sizeof(float)) / DOUBLE_BUFFER;
    inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize, blockSize) / sizePerElem;
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);

    foreachSoloTilingData_.set_needCoreNum(numBlocks_);
    foreachSoloTilingData_.set_totalTensorCount(totalTensorCount_);
    foreachSoloTilingData_.set_tensorDataCountList(tensorDataCountList_);
    foreachSoloTilingData_.set_tensorStartList(tensorStartList_);
    foreachSoloTilingData_.set_tensorEndList(tensorEndList_);
    foreachSoloTilingData_.set_tensorStartOffsetList(tensorStartOffsetList_);
    foreachSoloTilingData_.set_tensorEndOffsetList(tensorEndOffsetList_);
    foreachSoloTilingData_.set_tensorMiddleCountList(tensorMiddleCountList_);
    foreachSoloTilingData_.set_tensorMiddleStartList(tensorMiddleStartList_);
    foreachSoloTilingData_.set_coreMiddleOffsetList(coreMiddleOffsetList_);
    foreachSoloTilingData_.set_maxTensorNumPerCore(maxTensorNumPerCore_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachReduceRegbaseTiling::PostTiling()
{
    context_->SetBlockDim(numBlocks_);
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动

    // WorkspaceSize
    size_t usrSize = (MAX_CORE_CONT_910D + MAX_TENSOR_CONT_910D) * sizeof(float);
    size_t sysWorkspaceSize = WORK_SPACE_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    auto tilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    foreachSoloTilingData_.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(foreachSoloTilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachReduceRegbaseTiling::GetTilingKey() const
{
    switch (dataType_) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT_950;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF_950;
        case ge::DT_BF16:
            return TILING_KEY_BF16_950;
        default:
            return 0;
    }
}

REGISTER_OPS_TILING_TEMPLATE(ForeachNorm, ForeachReduceRegbaseTiling, 30000);
} // namespace optiling