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
 * \file group_norm_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORM_IDX_IN_X = 0;
static constexpr size_t GROUPNORM_IDX_OUT_Y = 0;
static constexpr size_t GROUPNORM_IDX_OUT_MEAN = 1;
static constexpr size_t GROUPNORM_IDX_OUT_VAR = 2;
static constexpr size_t NUMGROUPS_IDX = 0;
static constexpr size_t N_IDX = 0;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static ge::graphStatus GroupNormV2InferShape(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do GroupNormV2InferShape");

  // get input shapes
  const gert::Shape* x_shape = context->GetInputShape(GROUPNORM_IDX_IN_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  // get output shapes
  gert::Shape* y_shape = context->GetOutputShape(GROUPNORM_IDX_OUT_Y);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  gert::Shape* mean_shape = context->GetOutputShape(GROUPNORM_IDX_OUT_MEAN);
  OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
  gert::Shape* var_shape = context->GetOutputShape(GROUPNORM_IDX_OUT_VAR);
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
  if (x_shape->GetDim(0) == UNKNOWN_DIM_VALUE_) {
    OP_LOGD(context->GetNodeName(), "Input shape is -1, set mean shape to (-1)");
    mean_shape->AppendDim(UNKNOWN_DIM_VALUE_);
  } else if (IsUnknownRank(x_shape)) {
    OP_LOGD(context->GetNodeName(), "Input shape is -2, set mean shape to (-2)");
    mean_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
  } else {
    mean_shape->AppendDim(*num_groups * n_dim);
  }
  *var_shape = *mean_shape;

  OP_LOGD(context->GetNodeName(), "End to do GroupNormV2InferShape");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupNormV2InferDataType(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do GroupNormV2InferDataType");
  auto input_value_dtype = context->GetInputDataType(GROUPNORM_IDX_IN_X);
  context->SetOutputDataType(GROUPNORM_IDX_OUT_Y, input_value_dtype);
  context->SetOutputDataType(GROUPNORM_IDX_OUT_MEAN, input_value_dtype);
  context->SetOutputDataType(GROUPNORM_IDX_OUT_VAR, input_value_dtype);
  OP_LOGD(context->GetNodeName(), "End to do GroupNormV2InferDataType");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormV2).InferShape(GroupNormV2InferShape).InferDataType(GroupNormV2InferDataType);
}  // namespace ops
