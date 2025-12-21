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
 * \file group_quant.cc
 * \brief
 */
#include <numeric>
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;

static ge::graphStatus InferDataType4GroupQuant(gert::InferDataTypeContext* context) {
  OP_CHECK_IF(context == nullptr,
           OP_LOGE("GroupQuant", "InferDataTypeContext is nullptr"),
           return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant begin");

  ge::DataType yDtype = ge::DT_INT8;
  const int32_t *pDstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
  if (pDstDtype != nullptr) {
    int32_t dstDtype = *pDstDtype;
    OP_CHECK_IF(dstDtype != DTYPE_INT8 && dstDtype != DTYPE_INT4,
             OP_LOGE("GroupQuant", "attr dst_type only support 2(int8) and 29(int4)"),
             return ge::GRAPH_FAILED);
    yDtype = static_cast<ge::DataType>(dstDtype);
  }
  context->SetOutputDataType(0, yDtype);

  OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant end");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GroupQuant(gert::InferShapeContext* context) {
  OP_CHECK_IF(context == nullptr,
           OP_LOGE("GroupQuant", "InferShapeContext is nullptr"),
           return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "InferShape4GroupQuant begin");

  const gert::Shape* xShape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

  gert::Shape* yShape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

  *yShape = *xShape;

  OP_LOGD(context->GetNodeName(), "InferShape4GroupQuant end. y shape:%s", Ops::Base::ToString(*yShape).c_str());
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupQuant).InferDataType(InferDataType4GroupQuant)
                              .InferShape(InferShape4GroupQuant);
}  // namespace ops