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
 * \file dynamic_block_quant.cc
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
constexpr size_t INPUT_DIM_NUM = 2;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

graphStatus InferShapeForDynamicBlockQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForDynamicBlockQuant");
    const gert::Shape* inputXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

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

    if (inputXShape->GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context, "only support input dim num is 2, infershape failed");
        return ge::GRAPH_FAILED;
    }

    int64_t dim0 = inputXShape->GetDim(0);
    int64_t dim1 = inputXShape->GetDim(1);

    OP_LOGD(
        context, "DynamicBlockQuant input shape is (%ld, %ld), rowBlockSize is %d, colBlockSize is %d",
        dim0, dim1, *rowBlockSize, *colBlockSize);

    *outputShape = *inputXShape;
    scaleShape->SetDimNum(INPUT_DIM_NUM);
    if (dim0 == UNKNOWN_DIM_VALUE) {
        scaleShape->SetDim(0, UNKNOWN_DIM_VALUE);
    } else {
        scaleShape->SetDim(0, Ops::Base::CeilDiv(dim0, static_cast<int64_t>(*rowBlockSize)));
    }
    if (dim1 == UNKNOWN_DIM_VALUE) {
        scaleShape->SetDim(1, UNKNOWN_DIM_VALUE);
    } else {
        scaleShape->SetDim(1, Ops::Base::CeilDiv(dim1, static_cast<int64_t>(*colBlockSize)));
    }

    OP_LOGD(context, "End to do InferShapeForDynamicBlockQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicBlockQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForDynamicBlockQuant");
    const int32_t* dstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context, "End to do InferDataTypeForDynamicBlockQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicBlockQuant)
    .InferShape(InferShapeForDynamicBlockQuant)
    .InferDataType(InferDataTypeForDynamicBlockQuant);
} // namespace ops