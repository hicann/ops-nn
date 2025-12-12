/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ge_glu_grad_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
namespace geglugradv2 {

constexpr size_t GE_GLU_GRAD_IN_DY = 0;
constexpr size_t GE_GLU_GRAD_IN_X = 1;
constexpr size_t GE_GLU_GRAD_IN_GELU = 2;
constexpr size_t GE_GLU_GRAD_OUT_DX = 0;
constexpr size_t GE_GLU_ATTR_DIM = 0;
constexpr int64_t GE_GLU_NUM_TWO = 2;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGeGluGradV2(gert::InferShapeContext* context)
{
    auto dy_shape = context->GetInputShape(GE_GLU_GRAD_IN_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context, dy_shape);
    auto x_shape = context->GetInputShape(GE_GLU_GRAD_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto gelu_shape = context->GetInputShape(GE_GLU_GRAD_IN_GELU);
    OP_CHECK_NULL_WITH_CONTEXT(context, gelu_shape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GE_GLU_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    auto split_dim = *split_dim_ptr;
    size_t dy_dim_num = dy_shape->GetDimNum();
    size_t x_dim_num = x_shape->GetDimNum();
    size_t gelu_dim_num = gelu_shape->GetDimNum();
    OP_CHECK_IF(
        x_dim_num == 0 || dy_dim_num == 0 || gelu_dim_num == 0,
        OP_LOGE("GEGLUGRADV2", "not support inputs have scalar, check failed"), return GRAPH_FAILED);

    OP_CHECK_IF(
        Ops::Base::IsUnknownRank(*x_shape) || Ops::Base::IsUnknownRank(*dy_shape) ||
            Ops::Base::IsUnknownRank(*gelu_shape),
        OP_LOGD("GEGLUGRADV2", "inputs have unknown rank, no need check."),
        return CopyShapeInput2OutputWithIdx(context, 1, 0));

    OP_CHECK_IF(
        x_dim_num != dy_dim_num || x_dim_num != gelu_dim_num,
        OP_LOGE("GEGLUGRADV2", "inputs shape is different, shape check fail."), return GRAPH_FAILED);

    if (split_dim < 0) {
        split_dim += x_dim_num;
    }
    OP_CHECK_IF(
        (split_dim < 0 || split_dim >= static_cast<int64_t>(x_dim_num)),
        OP_LOGE(
            "GEGLUGRADV2", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].", x_dim_num,
            x_dim_num - 1, split_dim),
        return GRAPH_FAILED);

    for (size_t i = 0; i < x_dim_num; i++) {
        int64_t dim_value_x = x_shape->GetDim(i);
        int64_t dim_value_dy = dy_shape->GetDim(i);
        int64_t dim_value_gelu = gelu_shape->GetDim(i);
        if (dim_value_x == UNKNOWN_DIM_VALUE_) {
            OP_CHECK_IF(
                (dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_gelu != UNKNOWN_DIM_VALUE_ &&
                 dim_value_dy != dim_value_gelu),
                OP_LOGE("GEGLUGRADV2", "The input shape dy and gelu is different, check failed."), return GRAPH_FAILED);
        } else {
            if (split_dim == static_cast<int64_t>(i)) {
                OP_CHECK_IF(
                    (dim_value_x % GE_GLU_NUM_TWO != 0),
                    OP_LOGE("GEGLUGRADV2", "The input x shape can not split by the attr dim, check failed."),
                    return GRAPH_FAILED);
                OP_CHECK_IF(
                    (dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_x != GE_GLU_NUM_TWO * dim_value_dy),
                    OP_LOGE("GEGLUGRADV2", "The input dy and x shape does not satisfy the operator constraint."),
                    return GRAPH_FAILED);
                OP_CHECK_IF(
                    (dim_value_gelu != UNKNOWN_DIM_VALUE_ && dim_value_x != GE_GLU_NUM_TWO * dim_value_gelu),
                    OP_LOGE("GEGLUGRADV2", "The input gelu and x shape does not satisfy the operator constraint."),
                    return GRAPH_FAILED);
            } else {
                OP_CHECK_IF(
                    (dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_x != dim_value_dy),
                    OP_LOGE("GEGLUGRADV2", "The input dy and x shape does not satisfy the operator constraint."),
                    return GRAPH_FAILED);
                OP_CHECK_IF(
                    (dim_value_gelu != UNKNOWN_DIM_VALUE_ && dim_value_x != dim_value_gelu),
                    OP_LOGE("GEGLUGRADV2", "The input gelu and x shape does not satisfy the operator constraint."),
                    return GRAPH_FAILED);
            }
        }
    }

    return CopyShapeInput2OutputWithIdx(context, 1, 0);
}

IMPL_OP_INFERSHAPE(GeGluGradV2).InferShape(InferShapeForGeGluGradV2);
} // namespace geglugradv2
} // namespace ops
