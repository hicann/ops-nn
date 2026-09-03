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
 * \file avg_pool3_d_graph_infer.cpp
 * \brief Data type inference implementation for AvgPool3D.
 */

#include <cstddef>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_Y_INDEX = 0;

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

ge::graphStatus InferDataTypeAvgPool3D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeAvgPool3D.");

    const ge::DataType inputDtype = context->GetInputDataType(INPUT_X_INDEX);

    if (!IsSupportedDataType(inputDtype)) {
        OP_LOGE(context->GetNodeName(), "x must use a dtype supported by AvgPool3D.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_Y_INDEX, inputDtype);

    OP_LOGD(context->GetNodeName(), "End InferDataTypeAvgPool3D.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AvgPool3D).InferDataType(InferDataTypeAvgPool3D);

} // namespace ops
