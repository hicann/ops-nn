/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_offset_infershape.cpp
 * \brief
 */


#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ConcatOffset(gert::InferShapeContext* context) {
  // get shape of the first input from dynamic x
  constexpr size_t dynamic_x = 1;
  const gert::Shape* x_shape_0 = context->GetDynamicInputShape(dynamic_x, 0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape_0);

  // get dynamic num from attr N
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* dynamic_num = attrs->GetAttrPointer<int64_t>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dynamic_num);

  // get input num
  auto computeNodeInfo = context->GetComputeNodeInfo();
  OPS_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
  auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(dynamic_x);
  OPS_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
  uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
  OP_CHECK_IF(inputNum != *dynamic_num,
           OP_LOGE(context->GetNodeName(), "attr N should be same as input x tensor num"),
           return GRAPH_FAILED);

  for (int64_t i = 0; i < *dynamic_num; i++) {
    gert::Shape* out_shape = context->GetOutputShape(i);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *x_shape_0;
  }

  return ge::GRAPH_SUCCESS;
}


ge::graphStatus InferDataType4ConcatOffset(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4ConcatOffset start");
  int64_t totalOutNum = context->GetComputeNodeOutputNum();

  for (int64_t i = 0; i < totalOutNum; i++) {
      context->SetOutputDataType(i, DT_INT32);   
  }
  OP_LOGD(context->GetNodeName(), "InferDataType4ConcatOffset end");
  return GRAPH_SUCCESS;
}


IMPL_OP_INFERSHAPE(ConcatOffset).InferShape(InferShape4ConcatOffset).InferDataType(InferDataType4ConcatOffset);
}  // namespace ops
