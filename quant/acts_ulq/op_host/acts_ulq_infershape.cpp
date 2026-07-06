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

static bool BroadcastInferShape3(const gert::Shape& s0, const gert::Shape& s1, const gert::Shape& s2, gert::Shape& out)
{
    int64_t r0 = s0.GetDimNum(), r1 = s1.GetDimNum(), r2 = s2.GetDimNum();
    int64_t maxR = std::max({r0, r1, r2});
    if (maxR > 8)
        return false;

    std::vector<int64_t> result(maxR, 1);
    auto merge = [&](const gert::Shape& s) {
        int64_t r = s.GetDimNum();
        int64_t offset = maxR - r;
        for (int64_t d = 0; d < r; d++) {
            int64_t dim = s.GetDim(d);
            int64_t cur = result[d + offset];
            if (cur == 1)
                result[d + offset] = dim;
            else if (dim != 1 && dim != cur)
                return false;
        }
        return true;
    };
    if (!merge(s0) || !merge(s1) || !merge(s2))
        return false;

    out.SetDimNum(maxR);
    for (int64_t d = 0; d < maxR; d++)
        out.SetDim(d, result[d]);
    return true;
}

static ge::graphStatus InferShape4ActsUlq(gert::InferShapeContext* context)
{
    const gert::Shape* data_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_shape);
    const gert::Shape* cmin_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmin_shape);
    const gert::Shape* cmax_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmax_shape);

    gert::Shape out_shape;
    OP_CHECK_IF(!BroadcastInferShape3(*data_shape, *cmin_shape, *cmax_shape, out_shape),
                OP_LOGE(context, "ActsULQ: broadcast shape incompatible"), return ge::GRAPH_FAILED);

    for (int i = 0; i < 4; i++) {
        gert::Shape* output_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
        *output_shape = out_shape;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ActsULQ).InferShape(InferShape4ActsUlq);

} // namespace ops
