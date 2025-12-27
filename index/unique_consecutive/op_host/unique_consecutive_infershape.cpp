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
 * \file unique_consecutive_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static constexpr size_t OUT_IDX = 0;
static constexpr size_t INDICES_IDX = 1;
static constexpr size_t COUNTS_DIX = 2;
static constexpr size_t RETURN_IDX_IDX = 0;
static constexpr size_t REUTRN_COUNTS_IDX = 1;
static constexpr size_t AXIS_IDX = 2;
static constexpr int64_t MAX_VALID_AXIS = 8;
static ge::graphStatus InferShape4UniqueConsecutive(gert::InferShapeContext* context) {
    OP_LOGD(context->GetNodeName(), "InferShape4UniqueConsecutive running begin");
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* out_shape = context->GetOutputShape(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    gert::Shape* indices_shape = context->GetOutputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    gert::Shape* counts_shape = context->GetOutputShape(COUNTS_DIX);
    OP_CHECK_NULL_WITH_CONTEXT(context, counts_shape);
    if (IsUnknownRank(*x_shape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set shape to (-2)");
        SetUnknownRank(*out_shape);
        SetUnknownRank(*indices_shape);
        SetUnknownRank(*counts_shape);
        return GRAPH_SUCCESS;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *axis = attrs->GetAttrPointer<int64_t>(AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    size_t xDimNum = x_shape->GetDimNum();
    int64_t xSize = x_shape->GetShapeSize();
    OP_LOGW("InferShape4UniqueConsecutive", "Get x size = %ld", xSize);

    if (*axis <= MAX_VALID_AXIS) {
        out_shape->SetDimNum(xDimNum);
        SetUnknownShape(xDimNum, *out_shape);
        indices_shape->SetDimNum(1);
        indices_shape->SetDim(0, -1);
        counts_shape->SetDimNum(1);
        counts_shape->SetDim(0, -1);
    } else {
        if (xSize == 0) {
            OP_LOGW("InferShape4UniqueConsecutive", "Empty tensor.");
            *indices_shape = *x_shape;
            out_shape->SetDimNum(1);
            out_shape->SetDim(0, 0);
            counts_shape->SetDimNum(1);
            counts_shape->SetDim(0, 0);
        } else {
            indices_shape->SetDimNum(xDimNum);
            SetUnknownShape(xDimNum, *indices_shape);
            out_shape->SetDimNum(1);
            out_shape->SetDim(0, -1);
            counts_shape->SetDimNum(1);
            counts_shape->SetDim(0, -1);
        }
    }

    OP_LOGD(context->GetNodeName(), "InferShape4UniqueConsecutive running end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4UniqueConsecutive(gert::InferDataTypeContext* context) {
    OP_LOGD(context->GetNodeName(), "InferDtype4UniqueConsecutive begin");
    auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(OUT_IDX, input_dtype);
    context->SetOutputDataType(INDICES_IDX, DT_INT64);
    context->SetOutputDataType(COUNTS_DIX, DT_INT64);
    OP_LOGD(context->GetNodeName(), "OutputDtype: %s", ToString(context->GetOutputDataType(OUT_IDX)).c_str());
    OP_LOGD(context->GetNodeName(), "IdxDtype: %s", ToString(context->GetOutputDataType(INDICES_IDX)).c_str());
    OP_LOGD(context->GetNodeName(), "CountsDtype: %s", ToString(context->GetOutputDataType(COUNTS_DIX)).c_str());
    OP_LOGD(context->GetNodeName(), "InferDtype4UniqueConsecutive end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4UniqueConsecutive(gert::InferShapeRangeContext* context) {
    auto x_range = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_range);
    auto out_range = context->GetOutputShapeRange(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range);
    auto indices_range = context->GetOutputShapeRange(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_range);
    auto counts_range = context->GetOutputShapeRange(COUNTS_DIX);
    OP_CHECK_NULL_WITH_CONTEXT(context, counts_range);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *axis = attrs->GetAttrPointer<int64_t>(AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);
    auto real_axis = *axis >= 0 ? *axis : *axis + x_range->GetMax()->GetDimNum();

    OP_CHECK_NULL_WITH_CONTEXT(context, indices_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, out_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, counts_range->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, counts_range->GetMin());

    if (real_axis <= MAX_VALID_AXIS) {
        out_range->GetMax()->SetDimNum(x_range->GetMax()->GetDimNum());
        out_range->GetMin()->SetDimNum(x_range->GetMin()->GetDimNum());
        for (size_t i = 0; i < x_range->GetMax()->GetDimNum(); ++i) {
            out_range->GetMax()->SetDim(i, x_range->GetMax()->GetDim(i));
            out_range->GetMin()->SetDim(i, 0);
        }

        indices_range->GetMax()->SetDimNum(1);
        indices_range->GetMin()->SetDimNum(1);
        indices_range->GetMax()->SetDim(0, x_range->GetMax()->GetDim(real_axis));
        indices_range->GetMin()->SetDim(0, 0);
    } else {
        indices_range->GetMax()->SetDimNum(x_range->GetMax()->GetDimNum());
        indices_range->GetMin()->SetDimNum(x_range->GetMin()->GetDimNum());
        for (size_t i = 0; i < x_range->GetMax()->GetDimNum(); ++i) {
            indices_range->GetMax()->SetDim(i, x_range->GetMax()->GetDim(i));
            indices_range->GetMin()->SetDim(i, 0);
        }

        out_range->GetMax()->SetDimNum(1);
        out_range->GetMin()->SetDimNum(1);
        out_range->GetMax()->SetDim(0, x_range->GetMax()->GetShapeSize());
        out_range->GetMin()->SetDim(0, 0);
    }

    counts_range->GetMax()->SetDimNum(1);
    counts_range->GetMin()->SetDimNum(1);
    auto range_max = real_axis <= MAX_VALID_AXIS ? x_range->GetMax()->GetDim(real_axis) :
                                                   x_range->GetMax()->GetShapeSize();
    counts_range->GetMax()->SetDim(0, range_max);
    counts_range->GetMin()->SetDim(0, 0);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UniqueConsecutive).InferShape(InferShape4UniqueConsecutive)
                          .InferDataType(InferDtype4UniqueConsecutive)
                          .InferShapeRange(InferShapeRange4UniqueConsecutive)
                          .OutputShapeDependOnCompute({OUT_IDX, INDICES_IDX, COUNTS_DIX});

}  // namespace ops
