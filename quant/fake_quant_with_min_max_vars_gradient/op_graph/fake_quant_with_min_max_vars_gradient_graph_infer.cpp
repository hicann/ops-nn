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
 * \file fake_quant_with_min_max_vars_gradient_graph_infer.cpp
 * \brief FakeQuantWithMinMaxVarsGradient graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
namespace ops {
static constexpr int64_t IDX_OUT_0 = 0;
static constexpr int64_t IDX_OUT_1 = 1;
static constexpr int64_t IDX_OUT_2 = 2;

ge::graphStatus InferDataTypeFakeQuantWithMinMaxVarsGradient(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeFakeQuantWithMinMaxVarsGradient");
    context->SetOutputDataType(IDX_OUT_0, ge::DT_FLOAT);
    context->SetOutputDataType(IDX_OUT_1, ge::DT_FLOAT);
    context->SetOutputDataType(IDX_OUT_2, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeFakeQuantWithMinMaxVarsGradient");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FakeQuantWithMinMaxVarsGradient).InferDataType(InferDataTypeFakeQuantWithMinMaxVarsGradient);
} // namespace ops
