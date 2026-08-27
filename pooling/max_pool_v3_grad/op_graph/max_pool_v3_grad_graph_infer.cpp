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
 * \file max_pool_v3_grad_graph_infer.cpp
 * \brief Data type inference implementation for MaxPoolV3Grad.
 */

#include <cstddef>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {

constexpr size_t ORIG_INPUT_INDEX = 0;
constexpr size_t ORIG_OUTPUT_INDEX = 1;
constexpr size_t GRAD_INDEX = 2;
constexpr size_t OUT_GRAD_INDEX = 0;
// todo:参照其他算子写
bool IsSupportedDataType(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_INT32:
        case ge::DT_INT64:
        case ge::DT_INT8:
        case ge::DT_UINT16:
        case ge::DT_UINT8:
        case ge::DT_BF16:
            return true;
        default:
            return false;
    }
}

ge::graphStatus InferDataTypeMaxPoolV3Grad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeMaxPoolV3Grad.");

    const ge::DataType origInputDtype = context->GetInputDataType(ORIG_INPUT_INDEX);
    const ge::DataType origOutputDtype = context->GetInputDataType(ORIG_OUTPUT_INDEX);
    const ge::DataType gradDtype = context->GetInputDataType(GRAD_INDEX);

    if (!IsSupportedDataType(origInputDtype) || !IsSupportedDataType(origOutputDtype) ||
        !IsSupportedDataType(gradDtype)) {
        OP_LOGE(context->GetNodeName(), "orig_input, orig_output and grad must use a dtype supported "
                                        "by MaxPoolV3Grad.");
        return ge::GRAPH_FAILED;
    }

    if (origInputDtype != origOutputDtype || origInputDtype != gradDtype) {
        OP_LOGE(context->GetNodeName(), "orig_input, orig_output and grad must have the same dtype.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUT_GRAD_INDEX, origInputDtype);

    OP_LOGD(context->GetNodeName(), "End InferDataTypeMaxPoolV3Grad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolV3Grad).InferDataType(InferDataTypeMaxPoolV3Grad);

} // namespace ops
