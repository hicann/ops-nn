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
 * \file apply_adagrad_infershape.cpp
 * \brief ApplyAdagrad infershape.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
inline ge::graphStatus CopyShapeInputToOutputWithIdx(gert::InferShapeContext* context, int64_t inputIdx,
                                                     int64_t outputIdx)
{
    auto inShape = context->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(outputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyAdagrad(gert::InferShapeContext* context)
{
    constexpr size_t inputVarIdx = 0;
    constexpr size_t outputVarIdx = 0;
    return CopyShapeInputToOutputWithIdx(context, inputVarIdx, outputVarIdx);
}

static ge::graphStatus InferDataType4ApplyAdagrad(gert::InferDataTypeContext* context)
{
    constexpr size_t inputVarIdx = 0;
    constexpr size_t outputVarIdx = 0;
    context->SetOutputDataType(outputVarIdx, context->GetInputDataType(inputVarIdx));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdagrad).InferShape(InferShape4ApplyAdagrad).InferDataType(InferDataType4ApplyAdagrad);

} // namespace ops
