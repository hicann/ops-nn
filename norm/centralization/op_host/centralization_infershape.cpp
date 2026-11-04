/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr int64_t kMaxRank = 8;
} // namespace

static ge::graphStatus InferShape4Centralization(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4Centralization.");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGD(context, "x is unknown rank, set output to unknown rank");
        return ge::GRAPH_SUCCESS;
    }
    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    if (xShape->IsScalar() || rank > kMaxRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                              "rank must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Centralization(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4Centralization.");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4Centralization(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeRange4Centralization.");
    const auto* xRange = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    auto* yRange = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
    if (xRange->GetMin() == nullptr || xRange->GetMax() == nullptr || yRange->GetMin() == nullptr ||
        yRange->GetMax() == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    *yRange->GetMin() = *xRange->GetMin();
    *yRange->GetMax() = *xRange->GetMax();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Centralization)
    .InferShape(InferShape4Centralization)
    .InferDataType(InferDataType4Centralization)
    .InferShapeRange(InferShapeRange4Centralization);
} // namespace ops
