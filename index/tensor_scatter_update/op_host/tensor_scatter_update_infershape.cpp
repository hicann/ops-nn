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
 * \file tensor_scatter_update_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr size_t kXInputIdx = 0U;
constexpr size_t kYOutputIdx = 0U;
} // namespace

// y takes the dtype of x, matching the source proto which set the output dtype from
// op.GetInputDescByName("x").GetDataType().
static graphStatus InferDataTypeForTensorScatterUpdate(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeForTensorScatterUpdate");
    context->SetOutputDataType(kYOutputIdx, context->GetInputDataType(kXInputIdx));
    OP_LOGD(context->GetNodeName(), "End InferDataTypeForTensorScatterUpdate");
    return GRAPH_SUCCESS;
}

// The source proto copies x's shape straight to y with no rank or dim check, which is exactly what
// InferShape4Elewise does on input 0 / output 0 (unknown rank in -> unknown rank out).
IMPL_OP_INFERSHAPE(TensorScatterUpdate)
    .InferShape(Ops::Base::InferShape4Elewise)
    .InferDataType(InferDataTypeForTensorScatterUpdate);
} // namespace ops
