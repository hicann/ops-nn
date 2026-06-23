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
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t X_INDEX = 0;
static constexpr int64_t Y_INDEX = 0;
static constexpr int64_t MEAN_INDEX = 1;
static constexpr int64_t RSTD_INDEX = 2;
static constexpr int64_t ATTR_NUM_GROUPS = 0;

static ge::graphStatus InferShapeGroupNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGroupNorm");
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetDimNum() < 2, OP_LOGE(context->GetNodeName(), "x rank should be >= 2"), return GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroupsAttr = attrs->GetInt(ATTR_NUM_GROUPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroupsAttr);
    OP_CHECK_IF(
        *numGroupsAttr <= 0, OP_LOGE(context->GetNodeName(), "num_groups should be greater than 0"),
        return GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    gert::Shape* meanShape = context->GetOutputShape(MEAN_INDEX);
    gert::Shape* rstdShape = context->GetOutputShape(RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);

    *yShape = *xShape;
    meanShape->SetDimNum(2);
    meanShape->SetDim(0, xShape->GetDim(0));
    meanShape->SetDim(1, *numGroupsAttr);
    *rstdShape = *meanShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeGroupNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNorm).InferShape(InferShapeGroupNorm);
} // namespace ops
