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
 * \file nll_loss_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;

namespace ops {
const int64_t INPUT_IDX_X = 0;
const int64_t INPUT_IDX_TARGET = 2;
const int64_t INPUT_SIZE_TWO = 2;
const int64_t INPUT_SIZE_FOUR = 4;
const int64_t NUMBER_TWO = 2;
const int64_t NUMBER_THREE = 3;
const int64_t IDX_ZERO = 0;

static ge::graphStatus InferShapeForNLLLossGrad(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto x_shape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  auto target_shape = context->GetInputShape(INPUT_IDX_TARGET);
  OP_CHECK_NULL_WITH_CONTEXT(context, target_shape);

  auto out_y_shape = context->GetOutputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_y_shape);

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const char* attr_reduction = attrs->GetAttrPointer<char>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, attr_reduction);

  if (Ops::Base::IsUnknownRank(*x_shape)) {
    OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
    Ops::Base::SetUnknownRank(*out_y_shape);
    return ge::GRAPH_SUCCESS;
  }
  else if (x_shape->GetDimNum() == 1) {
    // x is [C], y is [1]
    out_y_shape->SetDimNum(1);
    out_y_shape->SetDim(0, x_shape->GetDim(0));

    return GRAPH_SUCCESS;
  }
  else if (x_shape->GetDimNum() == INPUT_SIZE_TWO) {
    // x is [N, C], y is [N]
    out_y_shape->SetDimNum(INPUT_SIZE_TWO);
    out_y_shape->SetDim(0, x_shape->GetDim(0));
    out_y_shape->SetDim(1, x_shape->GetDim(1));

    return GRAPH_SUCCESS;
  }
  else if (x_shape->GetDimNum() == INPUT_SIZE_FOUR) {
    // x is [N, C, H, W], y is [N, H, W]
    out_y_shape->SetDimNum(INPUT_SIZE_FOUR);
    out_y_shape->SetDim(0, x_shape->GetDim(0));
    out_y_shape->SetDim(1, x_shape->GetDim(1));
    out_y_shape->SetDim(NUMBER_TWO, x_shape->GetDim(NUMBER_TWO));
    out_y_shape->SetDim(NUMBER_THREE, x_shape->GetDim(NUMBER_THREE));
    return GRAPH_SUCCESS;
  }

  OP_LOGE(context->GetNodeName(), "x_shape->GetDimNum() only support 1, 2 and 4");
  return GRAPH_FAILED;
}

graphStatus InferDtypeForNLLLossGrad(gert::InferDataTypeContext* context)
{
  OP_LOGI(context->GetNodeName(), "inferdtype is begin");
  auto x_dtype = context->GetInputDataType(IDX_ZERO);
  OP_CHECK_IF((x_dtype != ge::DT_FLOAT) && (x_dtype != ge::DT_FLOAT16) && (x_dtype != ge::DT_BF16),
             OP_LOGE(context->GetNodeName(), "x_dtype only support float, float16 and bfloat16"),
             return GRAPH_FAILED);

  context->SetOutputDataType(IDX_ZERO, x_dtype);

  OP_LOGI(context->GetNodeName(), "inferdtype is success");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NLLLossGrad)
    .InferShape(InferShapeForNLLLossGrad)
    .InferDataType(InferDtypeForNLLLossGrad);
}  // namespace ops
