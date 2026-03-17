/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
const size_t INPUT_IDX_X = 0;
const size_t INPUT_IDX_INDICES = 1;
const int64_t DIM_TWO = 2;

static ge::graphStatus InferShapeForEmbedding(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto xShape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  int64_t xDim = xShape->GetDimNum();
  OP_CHECK_IF(
        xDim != DIM_TWO,
        OP_LOGE(context->GetNodeName(), "x input dim should be 2."),
        return ge::GRAPH_FAILED);
  
  auto indiesShape = context->GetInputShape(INPUT_IDX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indiesShape);
  int64_t rankIndices = indiesShape->GetDimNum();

  auto outShape = context->GetOutputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
  outShape->SetDimNum(rankIndices + 1);

  for (int64_t i = 0; i < rankIndices; i++) {
    outShape->SetDim(i, indiesShape->GetDim(i));
  }

  for (int64_t i = 1; i < xDim; i++) {
    outShape->SetDim(i, xShape->GetDim(i));
  }
  OP_CHECK_IF(
      xShape->GetShapeSize() > INT32_MAX || outShape->GetShapeSize() > INT32_MAX,
      OP_LOGE(context->GetNodeName(), "input or output shape is too large."),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Embedding).InferShape(InferShapeForEmbedding);
}  // namespace ops