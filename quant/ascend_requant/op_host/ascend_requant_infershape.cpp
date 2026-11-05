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

using namespace ge;

namespace ops {

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
    const bool xUnknownRank = is_unknown_rank(*input_shape);
    const bool scaleUnknownRank = is_unknown_rank(*req_scale_shape);

    if (!xUnknownRank && (input_shape->GetDimNum() < 1U || input_shape->GetDimNum() > 8U)) {
        OP_LOGE(context->GetNodeName(), "rank(x)[%zu] must be in [1, 8].", input_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (!scaleUnknownRank && req_scale_shape->GetDimNum() > 8U) {
        OP_LOGE(context->GetNodeName(), "rank(req_scale)[%zu] must be in [0, 8].", req_scale_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (!xUnknownRank && !scaleUnknownRank && req_scale_shape->GetDimNum() > input_shape->GetDimNum()) {
        OP_LOGE(context->GetNodeName(),
                "rank(req_scale)[%zu] must be <= rank(x)[%zu], "
                "req_scale is not broadcastable to x (shape_mismatch).",
                req_scale_shape->GetDimNum(), input_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (!xUnknownRank && !scaleUnknownRank) {
        const size_t xRank = input_shape->GetDimNum();
        const size_t scaleRank = req_scale_shape->GetDimNum();
        if (xRank < 1U || scaleRank > xRank) {
            OP_LOGE(context->GetNodeName(),
                    "rank(req_scale)[%zu] must be in [0, rank(x)[%zu]], req_scale is not broadcastable to x.",
                    scaleRank, xRank);
            return ge::GRAPH_FAILED;
        }
        const auto is_unknown_dim = [](int64_t dim) -> bool { return dim == ge::UNKNOWN_DIM; };
        for (size_t i = 0; i < scaleRank; i++) {
            const int64_t xDim = input_shape->GetDim(xRank - scaleRank + i);
            const int64_t scaleDim = req_scale_shape->GetDim(i);
            if (is_unknown_dim(xDim) || is_unknown_dim(scaleDim)) {
                continue;
            }
            if (scaleDim != 1 && scaleDim != xDim) {
                OP_LOGE(context->GetNodeName(),
                        "req_scale dim[%zu]=%lld is neither 1 nor equal to x dim[%zu]=%lld; "
                        "req_scale must broadcast to x (broadcast_result_must_equal_x).",
                        i, scaleDim, xRank - scaleRank + i, xDim);
                return ge::GRAPH_FAILED;
            }
        }
    }

    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendRequant).InferShape(InferShape4AscendRequant);

} // namespace ops
