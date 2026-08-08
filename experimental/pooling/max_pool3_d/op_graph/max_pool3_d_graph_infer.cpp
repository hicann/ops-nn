/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_INDEX = 0U;
constexpr size_t OUTPUT_INDEX = 0U;

static ge::graphStatus InferDataTypeForMaxPool3DGraph(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MaxPool3D", "Graph infer-dtype context is null.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_INDEX, context->GetInputDataType(INPUT_INDEX));
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP(MaxPool3D).InferDataType(InferDataTypeForMaxPool3DGraph);
} // namespace ops
