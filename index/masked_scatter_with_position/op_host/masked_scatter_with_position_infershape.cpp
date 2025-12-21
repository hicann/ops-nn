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
 * \file masked_scatter_with_position_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeMaskedScatterWithPosition(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMaskedScatterWithPosition");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto xShapeSize = xShape->GetShapeSize();
    if (xShapeSize < 0) {
        return ge::GRAPH_FAILED;
    }
    size_t ShapeSize = static_cast<size_t>(xShapeSize);
    yShape->SetDimNum(ShapeSize);
    for (size_t i = 0U; i < ShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedScatterWithPosition).InferShape(InferShapeMaskedScatterWithPosition);
} // namespace ops