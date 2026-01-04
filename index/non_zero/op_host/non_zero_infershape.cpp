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
 * \file non_zero_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr size_t OUTPUT_SHAPE_RANK = 2;
}
namespace ops {
graphStatus InferShape4NonZero(gert::InferShapeContext* context)
{
    const gert::Shape* in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);

    gert::Shape* out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transpose = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, transpose);

    if (IsUnknownRank(*in_shape)) {
        SetUnknownRank(*out_shape);
        return ge::GRAPH_SUCCESS;
    }

    const size_t output_rank = 2;
    out_shape->SetDimNum(output_rank);

    if (*transpose) {
        out_shape->SetDim(0, static_cast<int64_t>(in_shape->GetDimNum()));
        out_shape->SetDim(1, UNKNOWN_DIM);
    } else {
        out_shape->SetDim(0, UNKNOWN_DIM);
        out_shape->SetDim(1, static_cast<int64_t>(in_shape->GetDimNum()));
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRange4NonZero(gert::InferShapeRangeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferShapeRange4NonZero running begin");
    auto in_range = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_range);
    auto out_range = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transpose = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, transpose);

    OP_CHECK_NULL_WITH_CONTEXT(context, in_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMin());

    int64_t total_num = in_range->GetMax()->GetShapeSize();
    int64_t input_dim = in_range->GetMax()->GetDimNum();
    if (*transpose) {
        out_range->GetMax()->SetDimNum(OUTPUT_SHAPE_RANK);
        out_range->GetMax()->SetDim(0, input_dim);
        out_range->GetMax()->SetDim(1, total_num);
        out_range->GetMin()->SetDimNum(OUTPUT_SHAPE_RANK);
        out_range->GetMin()->SetDim(0, input_dim);
        out_range->GetMin()->SetDim(1, 0);
    } else {
        out_range->GetMax()->SetDimNum(OUTPUT_SHAPE_RANK);
        out_range->GetMax()->SetDim(0, total_num);
        out_range->GetMax()->SetDim(1, input_dim);
        out_range->GetMin()->SetDimNum(OUTPUT_SHAPE_RANK);
        out_range->GetMin()->SetDim(0, 0);
        out_range->GetMin()->SetDim(1, input_dim);
    }

    OP_LOGD(context->GetNodeName(), "Get out_shape MAX %s.", ToString(*(out_range->GetMax())).c_str());
    OP_LOGD(context->GetNodeName(), "Get out_shape MIN %s.", ToString(*(out_range->GetMin())).c_str());
    OP_LOGI(context->GetNodeName(), "InferShapeRange4NonZero run success");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4NonZero(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDtype4NonZero enter");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_LOGI(context->GetNodeName(), "set output dtype: %s", ToString(outDtype).c_str());
    context->SetOutputDataType(0, outDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(NonZero)
    .InferShape(InferShape4NonZero)
    .InferShapeRange(InferShapeRange4NonZero)
    .InferDataType(InferDataType4NonZero);
} // namespace ops
