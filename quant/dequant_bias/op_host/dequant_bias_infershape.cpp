/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_bias.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t SIZE_1 = 1;
static constexpr int64_t SIZE_27 = 27;

static ge::graphStatus InferShape4DequantBias(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShape4AddRmsNorm");

  // get input shapes
  const gert::Shape* xShape = context->GetInputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

  // get output shapes
  gert::Shape* yShape = context->GetOutputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

  size_t xDimNum = xShape->GetDimNum();
  yShape->SetDimNum(DIM_TWO);
  for (size_t i = 0; i < DIM_TWO; i++) {
    int64_t dim = (xDimNum == 1U) ? -1 : xShape->GetDim(i);
    yShape->SetDim(i, dim);
  }

  OP_LOGD(context->GetNodeName(), "End to do InferShape4DequantBias");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType4DequantBias(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DequantBias");
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* output_dtype = attrs->GetAttrPointer<int64_t>(0);
  const int64_t outdtype = (output_dtype == nullptr) ? 0 : *output_dtype;
  if (outdtype == SIZE_1) {
    context->SetOutputDataType(0, DT_FLOAT16);
  } else if (outdtype == SIZE_27) {
    context->SetOutputDataType(0, DT_BF16);
  }
  OP_LOGD(context->GetNodeName(), "End to do InferDataType4DequantBias");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantBias).InferShape(InferShape4DequantBias).InferDataType(InferDataType4DequantBias);
}  // namespace ops