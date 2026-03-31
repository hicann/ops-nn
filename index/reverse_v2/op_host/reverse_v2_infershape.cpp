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
 * \file reverse.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t IN_X = 0;
static constexpr size_t IN_AXIS = 1;
static constexpr size_t OUT_Y = 0;

static ge::graphStatus CopyShapeInput2OutputWithIdxForReverseV2(
    gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ReverseV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "begin to do InferShape4ReverseV2 with CopyShapeInput2OutputWithIdx");
  return CopyShapeInput2OutputWithIdxForReverseV2(context, IN_X, OUT_Y);
}

IMPL_OP_INFERSHAPE(ReverseV2).InferShape(InferShape4ReverseV2).InputsDataDependency({IN_AXIS});
}  // namespace ops
