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
 * \file group_norm_swish_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMSWISH_IDX_IN_X = 0;
static constexpr size_t GROUPNORMSWISH_IDX_IN_GAMMA = 1;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_Y = 0;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_MEAN = 1;
static constexpr size_t GROUPNORMSWISH_IDX_OUT_VAR = 2;
static constexpr size_t NUMGROUPS_IDX = 0;
static constexpr size_t N_IDX = 0;

static ge::graphStatus GroupNormSwishInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do GroupNormSwishInferShape");

    // get input shapes
    const gert::Shape* x_shape = context->GetInputShape(GROUPNORMSWISH_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    // get output shapes
    gert::Shape* y_shape = context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* var_shape = context->GetOutputShape(GROUPNORMSWISH_IDX_OUT_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

    *y_shape = *x_shape;
    mean_shape->SetDimNum(0);

    // process attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* num_groups = attrs->GetAttrPointer<int64_t>(NUMGROUPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_groups);

    // update mean and var shape
    const int64_t n_dim = x_shape->GetDim(N_IDX);
    mean_shape->AppendDim(n_dim);
    mean_shape->AppendDim(*num_groups);
    *var_shape = *mean_shape;

    OP_LOGD(context, "End to do GroupNormSwishInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormSwishInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "GroupNormSwishInferDtype enter");

    // Get input tout
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputXDtype = context->GetInputDataType(GROUPNORMSWISH_IDX_IN_X);
    auto inputGammaDtype = context->GetInputDataType(GROUPNORMSWISH_IDX_IN_GAMMA);
    context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_Y, inputXDtype);
    context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_MEAN, inputGammaDtype);
    context->SetOutputDataType(GROUPNORMSWISH_IDX_OUT_VAR, inputGammaDtype);

    OP_LOGD(context, "GroupNormSwishInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSwish).InferShape(GroupNormSwishInferShape).InferDataType(GroupNormSwishInferDtype);
} // namespace ops
