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
 * \file broadcast_infer.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
constexpr size_t THREE_NUM_INPUT = 3;

static ge::graphStatus Infershape4BroadcastByThree(gert::InferShapeContext* context) {
  std::vector<const gert::Shape*> broadcastShapes;
  for (size_t i = 0; i < THREE_NUM_INPUT; ++i) {
    auto inShape = context->GetInputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    broadcastShapes.push_back(inShape);
  }
  auto outShape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

  OP_CHECK_IF(!BroadcastShape(broadcastShapes, outShape),
           OP_LOGE(context->GetNodeName(), "L1LossGrad BroadcastShape failed!"),
           return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeOutSameInput(gert::InferDataTypeContext *context) {
  auto xDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, xDtype);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(L1LossGrad).InferShape(Infershape4BroadcastByThree)
                              .InferDataType(InferDataTypeOutSameInput);
}