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
 * \file gelu_mul.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static constexpr size_t SIZE_2 = 2;

static ge::graphStatus InferShape4GeluMul(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GeluMul");

  // get input shapes
  auto xShape = context->GetInputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

  // get output shapes
  auto yShape = context->GetOutputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

  size_t xDimNum = xShape->GetDimNum();
  yShape->SetDimNum(xDimNum);

  *yShape = *xShape;
  yShape->SetDim(xDimNum - 1, xShape->GetDim(xDimNum - 1) / SIZE_2);

  OP_LOGD(context->GetNodeName(), "End to do InferShape4GeluMul");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GeluMul(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GeluMul");

  auto input_dtype = context->GetInputDataType(IDX_0);

  context->SetOutputDataType(IDX_0, input_dtype);

  OP_LOGD(context->GetNodeName(), "End to do InferDataType4GeluMul");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GeluMul).InferShape(InferShape4GeluMul).InferDataType(InferDataType4GeluMul);
}  // namespace ops