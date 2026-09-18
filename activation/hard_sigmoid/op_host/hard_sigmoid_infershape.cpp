/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/shape_util.h"

namespace ops {
static constexpr size_t MAX_DIM_NUM = 8;
static ge::graphStatus InferShape4HardSigmoid(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    if (Ops::Base::IsUnknownRank(*inputShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        inputShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "The dim num of input_x must be less than or equal to 8"),
        return ge::GRAPH_FAILED);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4HardSigmoid(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HardSigmoid).InferShape(InferShape4HardSigmoid).InferDataType(InferDataType4HardSigmoid);
} // namespace ops
