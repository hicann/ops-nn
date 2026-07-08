/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t GLU_GRAD_IN_GRAD_OUT = 0;
constexpr size_t GLU_GRAD_IN_SELF = 1;
constexpr size_t GLU_GRAD_OUT = 0;
constexpr size_t GLU_GRAD_ATTR_DIM = 0;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForGluGrad(gert::InferShapeContext* context)
{
    auto self_shape = context->GetInputShape(GLU_GRAD_IN_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, self_shape);
    auto out_shape = context->GetOutputShape(GLU_GRAD_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_GRAD_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    *out_shape = *self_shape;
    size_t self_dim_num = self_shape->GetDimNum();

    OP_CHECK_IF(
        self_dim_num == 0, OP_LOGE("GLUGrad", "input self dim num is %zu, not support scalar", self_dim_num),
        return GRAPH_FAILED);

    OP_CHECK_IF(
        Ops::Base::IsUnknownRank(*self_shape), OP_LOGD("GLUGrad", "input self is unknown rank, no need check."),
        return GRAPH_SUCCESS);

    auto split_dim = *split_dim_ptr;
    if (split_dim < 0) {
        split_dim += static_cast<int64_t>(self_dim_num);
    }

    OP_CHECK_IF(
        (split_dim < 0 || split_dim >= static_cast<int64_t>(self_dim_num)),
        OP_LOGE(
            "GLUGrad", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].", self_dim_num,
            self_dim_num - 1, split_dim),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GLUGrad).InferShape(InferShapeForGluGrad);
} // namespace ops
