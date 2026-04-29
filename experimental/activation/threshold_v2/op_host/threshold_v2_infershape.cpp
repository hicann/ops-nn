/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * \file threshold_v2_infershape.cpp
 * \brief ThresholdV2 shape inference (out.shape = self.shape)
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ThresholdV2(gert::InferShapeContext* context)
{
    const gert::Shape* self_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, self_shape);

    gert::Shape* out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *self_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ThresholdV2).InferShape(InferShape4ThresholdV2);

} // namespace ops
