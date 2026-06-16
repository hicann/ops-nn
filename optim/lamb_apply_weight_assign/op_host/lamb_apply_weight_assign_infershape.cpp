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
 * \file lamb_apply_weight_assign_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t UPDATE_IDX = 3;
constexpr size_t PARAM_IDX = 4;
constexpr size_t OUTPUT_IDX = 0;

static ge::graphStatus InferShape4LambApplyWeightAssign(gert::InferShapeContext* context)
{
    auto update_shape = context->GetInputShape(UPDATE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, update_shape);
    auto param_shape = context->GetInputShape(PARAM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, param_shape);
    auto output_shape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    OP_CHECK_IF(
        !BroadcastShape(update_shape, param_shape, output_shape),
        OP_LOGE(
            context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*update_shape).c_str(),
            ToString(*param_shape).c_str()),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LambApplyWeightAssign).InferShape(InferShape4LambApplyWeightAssign);
} // namespace ops
