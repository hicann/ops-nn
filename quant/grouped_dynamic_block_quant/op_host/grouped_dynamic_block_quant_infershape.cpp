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
constexpr size_t INPUT_DIM_TWO = 2;
constexpr size_t INPUT_DIM_THREE = 3;

graphStatus InferShapeForGroupedDynamicBlockQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForGroupedDynamicBlockQuant");
    const gert::Shape* inputXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

    const gert::Shape* inputGroupListShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGroupListShape);
    int64_t groupNum = inputGroupListShape->GetDim(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t* rowBlockSize = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_ROW_BLOCK_SIZE);
    const int32_t* colBlockSize = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_COL_BLOCK_SIZE);

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

    if (inputXShape->GetDimNum() == INPUT_DIM_TWO) {
        int64_t dim0 = inputXShape->GetDim(0);
        int64_t dim1 = inputXShape->GetDim(1);

        OP_LOGD(
            context, "GroupedDynamicBlockQuant input shape is (%ld, %ld), rowBlockSize is %d, colBlockSize is %d",
            dim0, dim1, *rowBlockSize, *colBlockSize);
        
        *outputShape = *inputXShape;
        scaleShape->SetDimNum(INPUT_DIM_TWO);
        // In practice, the number of rows in each group may not be divisible by rowBlockSize, so we allocate additional space for groupNum here.
        scaleShape->SetDim(0, dim0 / static_cast<int64_t>(*rowBlockSize) + groupNum);
        scaleShape->SetDim(1, Ops::Base::CeilDiv(dim1, static_cast<int64_t>(*colBlockSize)));
    } else if (inputXShape->GetDimNum() == INPUT_DIM_THREE) {
        int64_t dim0 = inputXShape->GetDim(0);
        int64_t dim1 = inputXShape->GetDim(1);
        int64_t dim2 = inputXShape->GetDim(2);

        OP_LOGD(
            context, "GroupedDynamicBlockQuant input shape is (%ld, %ld, %ld), rowBlockSize is %d, colBlockSize is %d",
            dim0, dim1, dim2, *rowBlockSize, *colBlockSize);
        
        *outputShape = *inputXShape;
        scaleShape->SetDimNum(INPUT_DIM_THREE);
        scaleShape->SetDim(0, dim0);
        // In practice, the number of rows in each group may not be divisible by rowBlockSize, so we allocate additional space for groupNum here.
        scaleShape->SetDim(1, dim1 / static_cast<int64_t>(*rowBlockSize) + groupNum);
        scaleShape->SetDim(INPUT_DIM_TWO, Ops::Base::CeilDiv(dim2, static_cast<int64_t>(*colBlockSize)));
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