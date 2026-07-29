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
 * \file apply_gradient_descent_graph_infer.cpp
 * \brief apply_gradient_descent graph infer resource (InferDataType).
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeApplyGradientDescent(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeApplyGradientDescent");

    // Output var shares the dtype of input var (inplace elementwise update).
    ge::DataType varDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, varDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeApplyGradientDescent");
    return GRAPH_SUCCESS;
}

IMPL_OP(ApplyGradientDescent).InferDataType(InferDataTypeApplyGradientDescent);

} // namespace ops
