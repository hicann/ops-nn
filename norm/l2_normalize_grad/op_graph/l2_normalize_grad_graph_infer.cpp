/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferDataType4L2NormalizeGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const DataType xDataType = context->GetInputDataType(0);
    if (xDataType != DT_FLOAT16 && xDataType != DT_FLOAT) {
        OP_LOGE(context, "L2NormalizeGrad input x dtype is not supported: %d.", static_cast<int32_t>(xDataType));
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, xDataType) != GRAPH_SUCCESS) {
        OP_LOGE(context, "Failed to set L2NormalizeGrad dx output dtype.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPL_OP(L2NormalizeGrad).InferDataType(InferDataType4L2NormalizeGrad);
} // namespace ops
