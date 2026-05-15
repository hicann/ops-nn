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
 * \file optimized_transducer_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

 static ge::graphStatus InferShapeOptimizedTransducer(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeOptimizedTransducer");

    const gert::Shape* logitsShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);

    gert::Shape* lossShape = context->GetOutputShape(0);
    gert::Shape* gradShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);

    lossShape->SetDimNum(1);
    lossShape->SetDim(0, logitsShape->GetDim(0));

    *gradShape = *logitsShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeOptimizedTransducer");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(OptimizedTransducer).InferShape(InferShapeOptimizedTransducer);
} // namespace ops
