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
#include <algorithm>

using namespace ge;

namespace ops {

static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;

static inline bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

static bool BroadcastCompatible(const gert::Shape& a, const gert::Shape& b)
{
    int64_t ra = a.GetDimNum();
    int64_t rb = b.GetDimNum();
    int64_t maxR = std::max(ra, rb);
    for (int64_t d = 0; d < maxR; d++) {
        int64_t dimA = (d < (maxR - ra)) ? 1 : a.GetDim(d - (maxR - ra));
        int64_t dimB = (d < (maxR - rb)) ? 1 : b.GetDim(d - (maxR - rb));
        if (dimA == -1 || dimB == -1)
            continue;
        if (dimA != dimB && dimA != 1 && dimB != 1)
            return false;
    }
    return true;
}

static ge::graphStatus InferShape4Dequantize(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* min_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, min_shape);
    const gert::Shape* max_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, max_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    *output_shape = *x_shape;

    // Unknown rank (-2): pass through, skip broadcast check
    OP_CHECK_IF(IsUnknownRank(x_shape),
                OP_LOGD(context, "Dequantize: input x is unknown rank [-2], skip broadcast check."),
                return ge::GRAPH_SUCCESS);

    OP_CHECK_IF(!BroadcastCompatible(*x_shape, *min_shape),
                OP_LOGE(context, "Dequantize: x and min_range shapes are not broadcast compatible"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!BroadcastCompatible(*x_shape, *max_shape),
                OP_LOGE(context, "Dequantize: x and max_range shapes are not broadcast compatible"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4Dequantize(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Dequantize).InferShape(InferShape4Dequantize).InferDataType(InferDtype4Dequantize);

} // namespace ops
