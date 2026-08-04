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
 * \file apply_adagrad_graph_infer.cpp
 * \brief ApplyAdagrad graph infer resource.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr size_t INPUT_VAR_INDEX = 0;
static constexpr size_t OUTPUT_VAR_INDEX = 0;

static ge::graphStatus InferDataTypeApplyAdagrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeApplyAdagrad");
    context->SetOutputDataType(OUTPUT_VAR_INDEX, context->GetInputDataType(INPUT_VAR_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeApplyAdagrad");
    return GRAPH_SUCCESS;
}

IMPL_OP(ApplyAdagrad).InferDataType(InferDataTypeApplyAdagrad);

} // namespace ops
