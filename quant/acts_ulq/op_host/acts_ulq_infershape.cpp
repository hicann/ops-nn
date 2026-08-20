/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include <algorithm>

using namespace ge;

namespace ops {

static bool IsUnknownShape(const gert::Shape& s)
{
    for (int64_t d = 0; d < s.GetDimNum(); d++) {
        if (s.GetDim(d) == -1 || s.GetDim(d) == -2)
            return true;
    }
    return false;
}

static int64_t GetShapeSize(const gert::Shape& s)
{
    int64_t size = 1;
    for (int64_t d = 0; d < s.GetDimNum(); d++) {
        int64_t dim = s.GetDim(d);
        if (dim < 0)
            return -1;
        size *= dim;
    }
    return size;
}

static ge::graphStatus InferShape4ActsULQ(gert::InferShapeContext* context)
{
    const gert::Shape* data_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_shape);
    const gert::Shape* cmin_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmin_shape);
    const gert::Shape* cmax_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmax_shape);

    OP_CHECK_IF(
        data_shape->GetDimNum() > 8,
        OP_LOGE(context, "ActsULQ: rank of x must be <= 8, but got %zu", static_cast<size_t>(data_shape->GetDimNum())),
        return ge::GRAPH_FAILED);

    bool is_dynamic_min = IsUnknownShape(*cmin_shape);
    OP_CHECK_IF(GetShapeSize(*cmin_shape) != 1 && !is_dynamic_min, OP_LOGE(context, "The size of clamp_min must be 1!"),
                return ge::GRAPH_FAILED);

    bool is_dynamic_max = IsUnknownShape(*cmax_shape);
    OP_CHECK_IF(GetShapeSize(*cmax_shape) != 1 && !is_dynamic_max, OP_LOGE(context, "The size of clamp_max must be 1!"),
                return ge::GRAPH_FAILED);

    for (int i = 0; i < 4; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
        *output_shape = *data_shape;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ActsULQ).InferShape(InferShape4ActsULQ);

} // namespace ops
