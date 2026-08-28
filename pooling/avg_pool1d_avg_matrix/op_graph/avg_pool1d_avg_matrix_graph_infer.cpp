/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
ge::graphStatus InferDataTypeForAvgPool1DAvgMatrix(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto status = context->SetOutputDataType(0, context->GetInputDataType(0));
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Set output data type failed");
    }
    return status;
}

IMPL_OP(AvgPool1DAvgMatrix).InferDataType(InferDataTypeForAvgPool1DAvgMatrix);
} // namespace ops
