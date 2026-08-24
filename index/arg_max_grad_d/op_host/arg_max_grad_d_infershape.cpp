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
 * \file arg_max_grad_d_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr uint64_t INPUT_VAR_IDX = 0;
constexpr uint64_t OUTPUT_Y_IDX = 0;
} // namespace

namespace ops {
// y 与 var 同形同 dtype(A2 的 ArgMaxGradDInferShape 同语义): 本算子只做按轴条件选择, 不改变形状
static graphStatus InferShape4ArgMaxGradD(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4ArgMaxGradD.");
    const gert::Shape* varShape = context->GetInputShape(INPUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*varShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    *yShape = *varShape;

    OP_LOGD(context, "InferShape4ArgMaxGradD End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4ArgMaxGradD(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataType4ArgMaxGradD Begin.");
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_VAR_IDX));
    OP_LOGD(context, "InferDataType4ArgMaxGradD End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ArgMaxGradD).InferShape(InferShape4ArgMaxGradD).InferDataType(InferDataType4ArgMaxGradD);
} // namespace ops
