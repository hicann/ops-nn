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
 * \file aclnn_embedding_bag.cpp
 * \brief
 */
 
#include "opdev/aicpu/aicpu_task.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "embedding_bag.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "op_api/aclnn_util.h"

namespace l0op {

OP_TYPE_REGISTER(EmbeddingBag);

static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

//根据dtype判断算子是否支持aicore
static inline bool IsAiCoreSupport(
    const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets, const aclTensor* perSampleWeights)
{
    if (perSampleWeights != nullptr && perSampleWeights->GetDataType() != weight->GetDataType()) {
        return false;
    }
    return op::CheckType(weight->GetDataType(), WEIGHT_DTYPE_SUPPORT_LIST) &&
           op::CheckType(indices->GetDataType(), INT_DTYPE_SUPPORT_LIST) &&
           op::CheckType(offsets->GetDataType(), INT_DTYPE_SUPPORT_LIST);
}

static op::Shape GetOutPutShape(
    const aclTensor* weight, const aclTensor* offsets, bool includeLastOffset)
{
    op::Shape outputShape;
    int64_t offsetSize = offsets->GetViewShape().GetDim(0);
    if (includeLastOffset) {
        offsetSize -= 1;
    }
    outputShape.AppendDim(offsetSize);
    outputShape.AppendDim((weight->GetViewShape())[1]);
    return outputShape;
}

static aclTensor* AllocTensorForEmbeddingBag(
    const op::Shape& shape910,
    op::DataType dtype910,
    const op::Shape& shapeOther,
    op::DataType dtypeOther,
    op::SocVersion socVersion,
    aclOpExecutor* executor)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return executor->AllocTensor(shape910, dtype910, op::Format::FORMAT_ND);
    } else {
        return executor->AllocTensor(shapeOther, dtypeOther, op::Format::FORMAT_ND);
    }
}

static std::pair<op::Shape, op::Shape> GetMaxIndicesShapes(
    const std::string& modeStr,
    bool includeLastOffset,
    const op::Shape& offsetsShape,
    const op::Shape& outputShape)
{
    op::Shape shape910;
    op::Shape shapeOther;

    if (modeStr == "max") {
        shape910 = outputShape;
        shapeOther = outputShape;
    } else {
        shape910.AppendDim(0);
        if (includeLastOffset) {
            shapeOther.AppendDim(offsetsShape.GetDim(0) - 1);
        } else {
            shapeOther = offsetsShape;
        }
    }
    return std::make_pair(shape910, shapeOther);
}

static std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> MakeNullEmbeddingBagResult() {
    return std::make_tuple(
        static_cast<aclTensor*>(nullptr),
        static_cast<aclTensor*>(nullptr),
        static_cast<aclTensor*>(nullptr),
        static_cast<aclTensor*>(nullptr));
}

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> EmbeddingBag(
    const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets, bool scaleGradByFreq,
    const std::string& modeStr, bool sparse, const aclTensor* perSampleWeights, bool includeLastOffset,
    int64_t paddingIdx, aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(weight, indices, offsets, perSampleWeights)) {
        return MakeNullEmbeddingBagResult();
    }
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    op::DataType indicesDtype = indices->GetDataType();
    op::DataType offsetsDtype = offsets->GetDataType();
    op::DataType indexPromoteDtype = (indicesDtype == offsetsDtype) ? indicesDtype : op::DataType::DT_INT64;
    //申请output_tensor的Tensor
    auto outputShape = GetOutPutShape(weight, offsets, includeLastOffset);
    auto outputTensor = executor->AllocTensor(outputShape, weight->GetDataType());
    aclTensor* offset2bag = nullptr;
    // 申请offset2bag的Tensor
    op::Shape offset2bagShape910, offset2bagShapeOther;
    offset2bagShape910.AppendDim(indices->GetViewShape().GetShapeSize());
    offset2bagShapeOther.AppendDim(indices->GetViewShape().GetShapeSize());
    offset2bag = AllocTensorForEmbeddingBag(offset2bagShape910, indexPromoteDtype, offset2bagShapeOther, indicesDtype, socVersion, executor);
    aclTensor* bagSize = nullptr;
    op::Shape bagSizeShape910, bagSizeShapeOther;
    bagSizeShape910.AppendDim(offsets->GetViewShape().GetDim(0));  
    if (includeLastOffset) {
        bagSizeShapeOther.AppendDim(offsets->GetViewShape().GetDim(0) - 1);
    } else {
        bagSizeShapeOther = offsets->GetViewShape();
    }
    bagSize = AllocTensorForEmbeddingBag(bagSizeShape910, indexPromoteDtype, bagSizeShapeOther, offsetsDtype, socVersion, executor);
    aclTensor* maxIndices = nullptr;
    auto maxIndicesShapes = GetMaxIndicesShapes(modeStr, includeLastOffset, offsets->GetViewShape(), outputShape);
    maxIndices = AllocTensorForEmbeddingBag(maxIndicesShapes.first, indexPromoteDtype, maxIndicesShapes.second, offsetsDtype, socVersion, executor);
    if (outputTensor == nullptr || offset2bag == nullptr || bagSize == nullptr || maxIndices == nullptr) {
        return MakeNullEmbeddingBagResult();
    }
    L0_DFX(
        EmbeddingBag, weight, indices, offsets, scaleGradByFreq, modeStr, sparse, perSampleWeights, includeLastOffset);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        EmbeddingBag, OP_INPUT(weight, indices, offsets, perSampleWeights),
        OP_OUTPUT(outputTensor, offset2bag, bagSize, maxIndices),
        OP_ATTR(modeStr, scaleGradByFreq, sparse, includeLastOffset, paddingIdx));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingBagAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return MakeNullEmbeddingBagResult();
    }
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(outputTensor, offset2bag, bagSize, maxIndices);
}
} // namespace l0op