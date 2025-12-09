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
 * \file masked_softmax_with_rel_pos_bias.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeMaskedSoftmaxWithRelPosBias(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter MaskedSoftmaxWithRelPosBias infershape impl.");
  // x shape
  const gert::Shape* x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  // y shape
  gert::Shape* y_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  *y_shape = *x_shape;
  OP_LOGI(context->GetNodeName(), "MaskedSoftmaxWithRelPosBias infershape end.");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMaskedSoftmaxWithRelPosBias(gert::InferDataTypeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter MaskedSoftmaxWithRelPosBias infershape impl.");
  auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedSoftmaxWithRelPosBias).InferShape(InferShapeMaskedSoftmaxWithRelPosBias).InferDataType(InferDataTypeMaskedSoftmaxWithRelPosBias);

}  // namespace ops