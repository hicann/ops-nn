/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
constexpr size_t INPUT_INDEX = 0U;
constexpr size_t OUTPUT_Y_INDEX = 0U;
constexpr size_t OUTPUT_ARGMAX_INDEX = 1U;
constexpr size_t ATTR_TARGMAX_INDEX = 3U;
constexpr int64_t TARGMAX_INT32 = 3;

ge::graphStatus InferDataTypeForMaxPoolWithArgmaxGraph(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MaxPoolWithArgmax", "Graph infer-dtype context is null.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_INDEX));

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE("MaxPoolWithArgmax", "Graph infer-dtype attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* targmax = attrs->GetAttrPointer<int64_t>(ATTR_TARGMAX_INDEX);
    if (targmax == nullptr) {
        OP_LOGE("MaxPoolWithArgmax", "Graph infer-dtype attr Targmax is null.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_ARGMAX_INDEX, *targmax == TARGMAX_INT32 ? ge::DT_INT32 : ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolWithArgmax).InferDataType(InferDataTypeForMaxPoolWithArgmaxGraph);
} // namespace ops
