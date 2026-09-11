/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {
static void SetNanMedianOutputShapes(gert::Shape& valuesShape, gert::Shape& indicesShape, const gert::Shape& inputShape,
                                     int64_t dim)
{
    valuesShape = inputShape;
    indicesShape = inputShape;
    valuesShape.SetDim(dim, 1);
    indicesShape.SetDim(dim, 1);
}

static ge::graphStatus GetNanMedianOutputShapes(gert::InferShapeContext* context, gert::Shape*& valuesShape,
                                                gert::Shape*& indicesShape)
{
    valuesShape = context->GetOutputShape(0);
    indicesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ResolveNanMedianDim(gert::InferShapeContext* context, int64_t rank, int64_t& normalizedDim)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    int64_t dim = dimAttr == nullptr ? -1 : *dimAttr;
    normalizedDim = dim < 0 ? dim + rank : dim;
    return normalizedDim >= 0 && normalizedDim < rank ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus NanMedianInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* valuesShape = nullptr;
    gert::Shape* indicesShape = nullptr;
    const ge::graphStatus outputStatus = GetNanMedianOutputShapes(context, valuesShape, indicesShape);
    if (outputStatus != ge::GRAPH_SUCCESS) {
        return outputStatus;
    }
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*valuesShape);
        Ops::Base::SetUnknownRank(*indicesShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t normDim = 0;
    if (ResolveNanMedianDim(context, rank, normDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetNanMedianOutputShapes(*valuesShape, *indicesShape, *xShape, normDim);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NanMedian).InferShape(NanMedianInferShape);
} // namespace ops
