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

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

namespace ops {
static ge::graphStatus KthValueInferShapeFunc(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto* valuesShape = context->GetOutputShape(0);
    auto* indicesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*valuesShape);
        Ops::Base::SetUnknownRank(*indicesShape);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(1);
    int64_t dim = (dimAttr == nullptr) ? -1 : *dimAttr;
    int64_t normDim = dim < 0 ? dim + rank : dim;
    if (normDim < 0 || normDim >= rank) {
        std::string dimValue = std::to_string(dim);
        std::string dimRange = "[" + std::to_string(-rank) + ", " + std::to_string(rank - 1) + "]";
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "dim", dimValue.c_str(), dimRange.c_str());
        return ge::GRAPH_FAILED;
    }
    *valuesShape = *xShape;
    *indicesShape = *xShape;
    valuesShape->SetDim(normDim, 1);
    indicesShape->SetDim(normDim, 1);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KthValue).InferShape(KthValueInferShapeFunc);
} // namespace ops
