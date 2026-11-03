/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_grad_v2_graph_infer.cpp
 * \brief GeluGradV2 graph InferDataType implementation
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

ge::graphStatus InferDataTypeGeluGradV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeGeluGradV2");
    // dy 与 x dtype 一致时输出随 x; 不一致(混合 dtype)时输出为 FLOAT, 与 aclnn 层 isMixDataType 契约一致
    const auto dyDtype = context->GetInputDataType(IDX_0);
    const auto xDtype = context->GetInputDataType(IDX_1);
    context->SetOutputDataType(IDX_0, (dyDtype == xDtype) ? xDtype : ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeGeluGradV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GeluGradV2).InferDataType(InferDataTypeGeluGradV2);
} // namespace ops
