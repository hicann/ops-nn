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
 * \file binary_cross_entropy_infershape.cpp
 * \brief binary_cross_entropy infershape
 */
#include "infershape_elewise_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4BinaryCrossEntropy(gert::InferShapeContext* context) {
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto reduction = attrs->GetAttrPointer<char>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
  if (strcmp(reduction, "none") == 0) {
    return Ops::Base::InferShape4Elewise(context);
  }
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  out_shape->SetDimNum(0);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BinaryCrossEntropy(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  OP_LOGD(context->GetNodeName(), "InferDataType4BinaryCrossEntropy enter");
  auto inputXDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputXDtype);
  OP_LOGD(context->GetNodeName(), "set BinaryCrossEntropy output dtype: %s", Ops::Base::ToString(inputXDtype).c_str());
  OP_LOGD(context->GetNodeName(), "InferDataType4BinaryCrossEntropy end");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BinaryCrossEntropy).InferShape(InferShape4BinaryCrossEntropy)
                                      .InferDataType(InferDataType4BinaryCrossEntropy);
}  // namespace ops