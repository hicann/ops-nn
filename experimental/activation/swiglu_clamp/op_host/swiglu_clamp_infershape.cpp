/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clamp_infershape.cpp
 * \brief SwigluClamp 形状推导: y shape = x shape,末维减半(gate/up 切分后取一半)
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeSwigluClamp(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSwigluClamp");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // y 继承 x 全部维度,末维减半
    *yShape = *xShape;
    const int64_t dimNum = xShape->GetDimNum();
    OP_CHECK_IF(dimNum < 1, OP_LOGE(context, "x dim num must be >= 1, got %ld", dimNum), return ge::GRAPH_FAILED);
    const int64_t lastDim = xShape->GetDim(dimNum - 1);
    OP_CHECK_IF(lastDim % 2 != 0, OP_LOGE(context, "swiglu_clamp: last dim must be even (2N), got %ld", lastDim),
                return ge::GRAPH_FAILED);
    yShape->SetDim(dimNum - 1, lastDim / 2);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSwigluClamp");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluClamp).InferShape(InferShapeSwigluClamp);
} // namespace ops
