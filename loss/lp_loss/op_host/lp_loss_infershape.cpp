/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_loss_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4LpLoss(gert::InferShapeContext *context) {
  gert::Shape* out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const char *reduction = attrs->GetAttrPointer<char>(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
  if (strcmp(reduction, "none") == 0) {
    // if reduction == "none" , output shape = x shape
    const gert::Shape* in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    const size_t input_rank = in_shape->GetDimNum();
    out_shape->SetDimNum(input_rank);
    for (size_t i = 0; i < input_rank; ++i) {
      out_shape->SetDim(i, in_shape->GetDim(i));
    }
  } else {
    // if reduction == "mean" or reduction == "sum" , output a scalar
    out_shape->SetDimNum(0);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LpLoss).InferShape(InferShape4LpLoss);
}  // namespace ops
