/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elu_grad_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

inline ge::graphStatus CopyShapeInputToOutputWithIdx(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx) {
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

namespace ops
{
constexpr size_t INPUT_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;

static ge::graphStatus InferShape4EluGradV2(gert::InferShapeContext* context)
{
    return CopyShapeInputToOutputWithIdx(context, INPUT_INDEX, OUTPUT_INDEX);
}

IMPL_OP_INFERSHAPE(EluGradV2).InferShape(InferShape4EluGradV2);
}  // namespace ops
