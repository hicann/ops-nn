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
 * \file mse_loss_v2_infershape.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_INDEX = 0;
static constexpr size_t TARGET_INDEX = 1;
static constexpr size_t OUTPUT_INDEX = 0;

graphStatus InferShapeForMSELossV2(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForMSELossV2.");
    const gert::Shape* input = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    const gert::Shape* target = context->GetInputShape(TARGET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, target);
    gert::Shape* output = context->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    std::string reduction = context->GetAttrs()->GetAttrPointer<char>(0);
    if (reduction == "none") {
        *output = *input;
    } else {
        context->GetOutputShape(OUTPUT_INDEX)->SetDimNum(0);
    }
    OP_LOGD(context, "End to do InferShapeForMSELossV2.");
    return GRAPH_SUCCESS;
}

graphStatus InferDataTypeForMSELossV2(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_INDEX, context->GetInputDataType(INPUT_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP(MSELossV2).InferShape(InferShapeForMSELossV2).InferDataType(InferDataTypeForMSELossV2);
} // namespace ops