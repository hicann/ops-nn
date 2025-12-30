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
 * \file scatter_add_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t SC_IN_VAR_IDX = 0;
static constexpr size_t SC_OUT_VAR_IDX = 0;

graphStatus InferDataType4ScatterAdd(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4ScatterAdd enter");
  auto input_x_dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, input_x_dtype);
  OP_LOGD(context->GetNodeName(), "InferDataType4ScatterAdd end");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterCommonInfer(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do ScatterAddInfershape.");
  const gert::Shape* var_shape = context->GetInputShape(SC_IN_VAR_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

  gert::Shape* output_shape = context->GetOutputShape(SC_OUT_VAR_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  if (Ops::Base::IsUnknownRank(*var_shape)) {
    OP_LOGD(context->GetNodeName(), "ScatterAdd input shape is UnknownRank, set output shape to (-2, )");
    Ops::Base::SetUnknownRank(*output_shape);
    return ge::GRAPH_SUCCESS;
  }

  *output_shape = *var_shape;

  OP_LOGD(context->GetNodeName(), "End to do ScatterAddInfershape.");

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterAdd).InferShape(InferShape4ScatterCommonInfer).InferDataType(InferDataType4ScatterAdd);

}  // namespace ops
