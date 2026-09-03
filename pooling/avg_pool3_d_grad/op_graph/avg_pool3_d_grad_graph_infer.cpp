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
 * \file avg_pool3_d_grad_graph_infer.cpp
 * \brief Data type inference implementation for AvgPool3DGrad.
 */

#include <cstddef>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t GRADS_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;

bool IsSupportedDataType(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return true;
        default:
            return false;
    }
}
} // namespace

ge::graphStatus InferDataTypeAvgPool3DGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeAvgPool3DGrad.");

    const ge::DataType gradsDtype = context->GetInputDataType(GRADS_INDEX);

    if (!IsSupportedDataType(gradsDtype)) {
        OP_LOGE(context->GetNodeName(), "grads must use a dtype supported by AvgPool3DGrad.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_INDEX, gradsDtype);

    OP_LOGD(context->GetNodeName(), "End InferDataTypeAvgPool3DGrad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AvgPool3DGrad).InferDataType(InferDataTypeAvgPool3DGrad);

} // namespace ops
