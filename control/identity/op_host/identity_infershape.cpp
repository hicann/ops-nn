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
 * \file identity_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"

using namespace gert;
namespace ops {
static ge::graphStatus InferShapeForIdentity(InferShapeContext *context) {
  auto input_num = context->GetComputeNodeInputNum();
  for (size_t i = 0U; i < input_num; ++i) {
    auto xshape = context->GetInputShape(i);
    auto yshape = context->GetOutputShape(i);
    if ((xshape == nullptr) || (yshape == nullptr)) {
      return ge::GRAPH_FAILED;
    }
    *yshape = *xshape;
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Identity).InferShape(InferShapeForIdentity);
}  // namespace ops

