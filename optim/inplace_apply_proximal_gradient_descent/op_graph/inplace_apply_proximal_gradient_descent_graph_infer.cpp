/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferDataTypeForInplaceApplyProximalGradientDescent(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType varDataType = context->GetInputDataType(0);
    if (varDataType != ge::DT_BF16 && varDataType != ge::DT_FLOAT16 && varDataType != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i < 5; ++i) {
        if (context->GetInputDataType(i) != varDataType) {
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, varDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(InplaceApplyProximalGradientDescent).InferDataType(InferDataTypeForInplaceApplyProximalGradientDescent);
} // namespace ops
