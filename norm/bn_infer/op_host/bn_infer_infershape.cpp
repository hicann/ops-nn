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
 * \file bn_infer_infershape.cpp
 * \brief BNInfer shape and dtype inference.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t Y_OUTPUT_IDX = 0;

static ge::graphStatus BNInferInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(Y_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus BNInferInferDataType(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(Y_OUTPUT_IDX, context->GetInputDataType(X_INPUT_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNInfer).InferShape(BNInferInferShape).InferDataType(BNInferInferDataType);
} // namespace ops
