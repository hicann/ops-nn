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
 * \file ge_glu_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_OUT_Y_GLU = 1;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForGeGluV2(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto out_shape_y = context->GetOutputShape(GLU_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y);
    auto out_shape_y_glu = context->GetOutputShape(GLU_OUT_Y_GLU);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y_glu);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    auto split_dim = *split_dim_ptr;
    *out_shape_y = *x_shape;
    *out_shape_y_glu = *x_shape;
    size_t x_dim_num = x_shape->GetDimNum();

    OP_CHECK_IF(
        x_dim_num == 0, OP_LOGE("GEGLUV2", "input x dim num is %zu, not support input x is scalar", x_dim_num),
        return GRAPH_FAILED);

    OP_CHECK_IF(
        Ops::Base::IsUnknownRank(*x_shape), OP_LOGD("GEGLUV2", "input x is unknown rank, no need check."),
        return GRAPH_SUCCESS);

    if (split_dim < 0) {
        split_dim += x_dim_num;
    }

    OP_CHECK_IF(
        (split_dim < 0 || split_dim >= static_cast<int64_t>(x_dim_num)),
        OP_LOGE(
            "GEGLUV2", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].", x_dim_num,
            x_dim_num - 1, split_dim),
        return GRAPH_FAILED);

    // 动态shape场景split_dim_value传入-1不做处理
    auto split_dim_value = x_shape->GetDim(split_dim);
    if (split_dim_value > 0) {
        out_shape_y->SetDim(split_dim, split_dim_value / SPLIT_NUM);
        out_shape_y_glu->SetDim(split_dim, split_dim_value / SPLIT_NUM);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GeGluV2).InferShape(InferShapeForGeGluV2);
} // namespace ops