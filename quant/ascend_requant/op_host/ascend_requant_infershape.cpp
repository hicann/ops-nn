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
#include "graph/operator_reg.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeAndType4AscendRequant(ge::Operator& op)
{
    const ge::TensorDesc x_desc = op.GetInputDesc(0);
    const ge::TensorDesc req_scale_desc = op.GetInputDesc(1);
    const std::vector<int64_t>& x_dims = x_desc.GetShape().GetDims();
    const std::vector<int64_t>& req_dims = req_scale_desc.GetShape().GetDims();

    const bool x_known = (x_dims != ge::UNKNOWN_RANK);
    const bool req_known = (req_dims != ge::UNKNOWN_RANK);
    if (x_known && req_known && req_dims.size() > x_dims.size()) {
        ge::AscendString op_name;
        (void)op.GetName(op_name);
        OP_LOGE(op_name.GetString(),
                "rank(req_scale)[%zu] must be <= rank(x)[%zu], "
                "req_scale is not broadcastable to x (shape_mismatch).",
                req_dims.size(), x_dims.size());
        return ge::GRAPH_FAILED;
    }

    ge::TensorDesc y_desc = op.GetOutputDesc(0);
    y_desc.SetShape(ge::Shape(x_dims));
    y_desc.SetOriginShape(ge::Shape(x_dims));
    y_desc.SetDataType(ge::DT_INT8);
    const ge::graphStatus ret = op.UpdateOutputDesc(0U, y_desc);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AscendRequant, InferShapeAndType4AscendRequant);

static ge::graphStatus InferShape4AscendRequant(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    const gert::Shape* req_scale_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, req_scale_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const auto is_unknown_rank = [](const gert::Shape& s) -> bool {
        return s.GetDimNum() == 1U && s.GetDim(0U) == ge::UNKNOWN_DIM_NUM;
    };
    if (!is_unknown_rank(*input_shape) && !is_unknown_rank(*req_scale_shape) &&
        req_scale_shape->GetDimNum() > input_shape->GetDimNum()) {
        OP_LOGE(context->GetNodeName(),
                "rank(req_scale)[%zu] must be <= rank(x)[%zu], "
                "req_scale is not broadcastable to x (shape_mismatch).",
                req_scale_shape->GetDimNum(), input_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendRequant).InferShape(InferShape4AscendRequant);

} // namespace ops
