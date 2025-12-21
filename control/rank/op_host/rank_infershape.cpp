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
 * \file rank_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"

using namespace gert;
namespace ops {
static ge::graphStatus InferShapeForRankLike(InferShapeContext *context) {
  auto x_shape = context->GetInputShape(0);
  auto y_shape = context->GetOutputShape(0);
  if ((x_shape == nullptr) || (y_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }

  y_shape->SetScalar();
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Rank).InferShape(InferShapeForRankLike);
IMPL_OP_INFERSHAPE(Size).InferShape(InferShapeForRankLike);
}  // namespace ops

