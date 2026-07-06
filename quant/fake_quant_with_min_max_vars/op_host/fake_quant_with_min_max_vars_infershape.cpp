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
 * \file fake_quant_with_min_max_vars_infershape.cpp
 * \brief FakeQuantWithMinMaxVars shape inference
 *
 * Rule: y.shape = x.shape (elementwise with per-tensor broadcast)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4FakeQuantWithMinMaxVars(gert::InferShapeContext* context)
{
    const gert::Shape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);

    gert::Shape* outputShapeY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapeY);

    // y.shape = x.shape (per-tensor broadcast, output mirrors input shape)
    *outputShapeY = *inputShapeX;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVars).InferShape(InferShape4FakeQuantWithMinMaxVars);

} // namespace ops