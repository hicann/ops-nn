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
static ge::graphStatus MedianInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* outputShapes[] = {context->GetOutputShape(0), context->GetOutputShape(1)};
    for (auto* outputShape : outputShapes) {
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    }
    if (Ops::Base::IsUnknownRank(*xShape)) {
        for (auto* outputShape : outputShapes) {
            Ops::Base::SetUnknownRank(*outputShape);
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    int64_t dim = dimAttr == nullptr ? -1 : *dimAttr;
    int64_t normDim = dim < 0 ? dim + rank : dim;
    if (normDim < 0 || normDim >= rank) {
        return ge::GRAPH_FAILED;
    }
    *outputShapes[0] = *xShape;
    *outputShapes[1] = *xShape;
    outputShapes[0]->SetDim(normDim, 1);
    outputShapes[1]->SetDim(normDim, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Median).InferShape(MedianInferShape);
} // namespace ops
