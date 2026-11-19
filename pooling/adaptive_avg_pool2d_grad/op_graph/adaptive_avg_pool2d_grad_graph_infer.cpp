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
 * \file adaptive_avg_pool2d_grad_graph_infer.cpp
 * \brief AdaptiveAvgPool2dGrad graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "error_util.h"
namespace ops {
static constexpr int64_t IDX_IN_0 = 0;
static constexpr int64_t IDX_OUT_0 = 0;

ge::graphStatus InferDataTypeAdaptiveAvgPool2dGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAdaptiveAvgPool2dGrad");
    const ge::DataType gradDtype = context->GetInputDataType(IDX_IN_0);
    if (gradDtype != ge::DT_FLOAT && gradDtype != ge::DT_FLOAT16 && gradDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input_grad data type",
                                  ge::TypeUtils::DataTypeToSerialString(gradDtype).c_str(), "float, float16, bfloat16");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(IDX_OUT_0, gradDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAdaptiveAvgPool2dGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AdaptiveAvgPool2dGrad).InferDataType(InferDataTypeAdaptiveAvgPool2dGrad);
} // namespace ops
