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
 * \file grouped_dynamic_block_quant.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_ATTR_DST_TYPE = 2;
constexpr size_t INDEX_ATTR_ROW_BLOCK_SIZE = 3;
constexpr size_t INDEX_ATTR_COL_BLOCK_SIZE = 4;
constexpr size_t INPUT_DIM_ONE = 1;
constexpr size_t INPUT_DIM_TWO = 2;
constexpr size_t INPUT_DIM_THREE = 3;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

void SetShapeDimTwo(gert::InferShapeContext* context, int64_t groupNum, int64_t rowBlockSize, int64_t colBlockSize)
{
    const gert::Shape* inputXShape = context->GetInputShape(0);
    const gert::Shape* inputGroupListShape = context->GetInputShape(1);
    gert::Shape* scaleShape = context->GetOutputShape(1);

    int64_t dim0 = inputXShape->GetDim(0);
    int64_t dim1 = inputXShape->GetDim(1);

    OP_LOGD(
        context, "GroupedDynamicBlockQuant input shape is [%s], rowBlockSize is %ld, colBlockSize is %ld", Ops::Base::ToString(*inputXShape).c_str(), rowBlockSize, colBlockSize);

    scaleShape->SetDimNum(INPUT_DIM_TWO);
    if (dim0 == UNKNOWN_DIM_VALUE || groupNum == UNKNOWN_DIM_VALUE || Ops::Base::IsUnknownRank(*inputGroupListShape)) {
        scaleShape->SetDim(0, UNKNOWN_DIM_VALUE);
    } else {
        if (rowBlockSize <= 0) {
            return;
        }
        // In practice, the number of rows in each group may not be divisible by rowBlockSize, so we allocate additional
        // space for groupNum here.
        scaleShape->SetDim(0, dim0 / rowBlockSize + groupNum);
    }
    if (dim1 == UNKNOWN_DIM_VALUE) {
        scaleShape->SetDim(1, UNKNOWN_DIM_VALUE);
    } else {
        scaleShape->SetDim(1, Ops::Base::CeilDiv(dim1, colBlockSize));
    }
}

void SetShapeDimThree(gert::InferShapeContext* context, int64_t groupNum, int64_t rowBlockSize, int64_t colBlockSize)
{
    const gert::Shape* inputXShape = context->GetInputShape(0);
    const gert::Shape* inputGroupListShape = context->GetInputShape(1);
    gert::Shape* scaleShape = context->GetOutputShape(1);

    int64_t dim0 = inputXShape->GetDim(0);
    int64_t dim1 = inputXShape->GetDim(1);
    int64_t dim2 = inputXShape->GetDim(2);

    OP_LOGD(
        context, "GroupedDynamicBlockQuant input shape is [%s], rowBlockSize is %ld, colBlockSize is %ld",
        Ops::Base::ToString(*inputXShape).c_str(), rowBlockSize, colBlockSize);

    scaleShape->SetDimNum(INPUT_DIM_THREE);
    if (dim0 == UNKNOWN_DIM_VALUE) {
        scaleShape->SetDim(0, UNKNOWN_DIM_VALUE);
    } else {
        scaleShape->SetDim(0, dim0);
    }
    if (dim1 == UNKNOWN_DIM_VALUE || groupNum == UNKNOWN_DIM_VALUE || Ops::Base::IsUnknownRank(*inputGroupListShape)) {
        scaleShape->SetDim(1, UNKNOWN_DIM_VALUE);
    } else {
        if (rowBlockSize <= 0) {
            return;
        }
        // In practice, the number of rows in each group may not be divisible by rowBlockSize, so we allocate additional
        // space for groupNum here.
        scaleShape->SetDim(1, dim1 / rowBlockSize + groupNum);
    }
    if (dim2 == UNKNOWN_DIM_VALUE) {
        scaleShape->SetDim(2, UNKNOWN_DIM_VALUE);
    } else {
        scaleShape->SetDim(2, Ops::Base::CeilDiv(dim2, colBlockSize));
    }
}

ge::graphStatus InferShapeForGroupedDynamicBlockQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForGroupedDynamicBlockQuant");
    const gert::Shape* inputXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

    const gert::Shape* inputGroupListShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGroupListShape);
    int64_t groupNum = inputGroupListShape->GetDim(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* rowBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ROW_BLOCK_SIZE);
    const int64_t* colBlockSize = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_COL_BLOCK_SIZE);

    if (rowBlockSize == nullptr || colBlockSize == nullptr) {
        OP_LOGD(context, "rowBlockSize or colBlockSize is nullptr, please check!");
        return ge::GRAPH_FAILED;
    }

    if ((*rowBlockSize) <= 0 || (*colBlockSize) <= 0) {
        OP_LOGD(context, "rowBlockSize or colBlockSize is invalid, please check!");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    gert::Shape* scaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    if (Ops::Base::IsUnknownRank(*inputXShape)) {
        OP_LOGD(context, "input shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*outputShape);
        Ops::Base::SetUnknownRank(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }

    *outputShape = *inputXShape;

    if (inputXShape->GetDimNum() == INPUT_DIM_TWO) {
        SetShapeDimTwo(context, groupNum, *rowBlockSize, *colBlockSize);
    } else if (inputXShape->GetDimNum() == INPUT_DIM_THREE) {
        SetShapeDimThree(context, groupNum, *rowBlockSize, *colBlockSize);
    } else {
        OP_LOGE(context, "only support input dim num is 2 or 3, infershape failed");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "End to do InferShapeForGroupedDynamicBlockQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForGroupedDynamicBlockQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForGroupedDynamicBlockQuant");
    const int32_t* dstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context, "End to do InferDataTypeForGroupedDynamicBlockQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedDynamicBlockQuant)
    .InferShape(InferShapeForGroupedDynamicBlockQuant)
    .InferDataType(InferDataTypeForGroupedDynamicBlockQuant);
} // namespace ops