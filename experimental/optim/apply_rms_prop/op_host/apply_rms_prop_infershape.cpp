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

/*!
 * \file apply_rms_prop_infershape.cpp
 * \brief apply_rms_prop 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t INPUT_IDX_VAR  = 0;
static constexpr size_t INPUT_IDX_MS   = 1;
static constexpr size_t INPUT_IDX_MOM  = 2;
static constexpr size_t INPUT_IDX_GRAD = 3;

static bool ShapeEqual(const gert::Shape& a, const gert::Shape& b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShape4ApplyRmsProp(gert::InferShapeContext* context)
{
    const gert::Shape* var_shape = context->GetInputShape(INPUT_IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

    const gert::Shape* ms_shape = context->GetInputShape(INPUT_IDX_MS);
    OP_CHECK_NULL_WITH_CONTEXT(context, ms_shape);
    OP_CHECK_IF(!ShapeEqual(*var_shape, *ms_shape),
        OP_LOGE(context, "ApplyRmsProp InferShape: ms shape must equal var shape"),
        return ge::GRAPH_FAILED);

    const gert::Shape* mom_shape = context->GetInputShape(INPUT_IDX_MOM);
    OP_CHECK_NULL_WITH_CONTEXT(context, mom_shape);
    OP_CHECK_IF(!ShapeEqual(*var_shape, *mom_shape),
        OP_LOGE(context, "ApplyRmsProp InferShape: mom shape must equal var shape"),
        return ge::GRAPH_FAILED);

    const gert::Shape* grad_shape = context->GetInputShape(INPUT_IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
    OP_CHECK_IF(!ShapeEqual(*var_shape, *grad_shape),
        OP_LOGE(context, "ApplyRmsProp InferShape: grad shape must equal var shape"),
        return ge::GRAPH_FAILED);

    gert::Shape* var_out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_out_shape);
    gert::Shape* ms_out_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ms_out_shape);
    gert::Shape* mom_out_shape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, mom_out_shape);

    *var_out_shape = *var_shape;
    *ms_out_shape  = *var_shape;
    *mom_out_shape = *var_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRmsProp).InferShape(InferShape4ApplyRmsProp);

}  // namespace ops
