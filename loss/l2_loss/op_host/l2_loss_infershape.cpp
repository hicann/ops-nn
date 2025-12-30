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
 * \file l2_loss.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
static ge::graphStatus InferShape4L2Loss(gert::InferShapeContext *context) {
  OP_LOGD(context->GetNodeName(), "Do InferShape4L2Loss start");

  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  out_shape->SetDimNum(0);

  OP_LOGD(context->GetNodeName(), "output y = %s", ToString(*out_shape).c_str());
  OP_LOGD(context->GetNodeName(), "Do InferShape4L2Loss success");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(L2Loss).InferShape(InferShape4L2Loss);
}  // namespace ops
