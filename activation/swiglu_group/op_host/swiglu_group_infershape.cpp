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
 * \file swiglu_group_infershape.cpp
 * \brief Shape and dtype inference for SwigluGroup.
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr int64_t NUM_TWO = 2;
} // namespace

graphStatus InferShape4SwigluGroup(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SwigluGroup.");
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t xRank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(xRank < 1,
                OP_LOGE(context->GetNodeName(), "The rank of x should be greater than 0, but is %ld.", xRank),
                return ge::GRAPH_FAILED);
    int64_t splitDim = xRank - 1;

    *yShape = *xShape;
    if (xShape->GetDim(splitDim) == -1) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % NUM_TWO != 0,
                OP_LOGE(context->GetNodeName(),
                        "The last dimension of x should be non-negative and divisible by 2, but got %ld.",
                        xShape->GetDim(splitDim)),
                return ge::GRAPH_FAILED);

    yShape->SetDim(splitDim, xShape->GetDim(splitDim) / NUM_TWO);

    OP_LOGD(context->GetNodeName(), "End to do InferShape4SwigluGroup.");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4SwigluGroup(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDtype4SwigluGroup.");
    auto xDtype = context->GetInputDataType(INPUT_IDX_X);
    context->SetOutputDataType(OUTPUT_IDX_Y, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDtype4SwigluGroup.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroup).InferShape(InferShape4SwigluGroup).InferDataType(InferDtype4SwigluGroup);
} // namespace ops
