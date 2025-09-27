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
 * \file squared_relu.cc
 * \brief
 */
 #include "register/op_impl_registry.h"
 #include "log/log.h"

 using namespace ge;

 namespace ops {

 static constexpr int64_t IDX_0 = 0;

 static ge::graphStatus InferShape4SquaredRelu(gert::InferShapeContext* context) {
   OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SquaredRelu");

   // get input shapes
   auto xShape = context->GetInputShape(IDX_0);
   OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

   // get output shapes
   auto yShape = context->GetOutputShape(IDX_0);
   OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

   size_t xDimNum = xShape->GetDimNum();
   yShape->SetDimNum(xDimNum);

   *yShape = *xShape;

   OP_LOGD(context->GetNodeName(), "End to do InferShape4SquaredRelu");
   return GRAPH_SUCCESS;
 }

 static graphStatus InferDataType4SquaredRelu(gert::InferDataTypeContext* context) {
   OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SquaredRelu");

   auto input_dtype = context->GetInputDataType(IDX_0);

   context->SetOutputDataType(IDX_0, input_dtype);

   OP_LOGD(context->GetNodeName(), "End to do InferDataType4SquaredRelu");

   return GRAPH_SUCCESS;
 }

 IMPL_OP_INFERSHAPE(SquaredRelu).InferShape(InferShape4SquaredRelu).InferDataType(InferDataType4SquaredRelu);
 }  // namespace ops