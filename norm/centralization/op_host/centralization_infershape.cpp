/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdint>
#include <string>

#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr int64_t kMaxRank = 8;
constexpr size_t kAttrAxesIdx = 0;

static ge::graphStatus ValidateDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType xDtype = context->GetInputDataType(0);
    if (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype),
                                  "FLOAT or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAxes(gert::InferShapeContext* context, int64_t rank)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* axesAttr = attrs->GetListInt(kAttrAxesIdx);
    if (axesAttr == nullptr || axesAttr->GetSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t* axesData = axesAttr->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, axesData);
    if (axesAttr->GetSize() > static_cast<size_t>(rank)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "axes", std::to_string(axesAttr->GetSize()).c_str(),
                                     std::to_string(rank).c_str());
        return ge::GRAPH_FAILED;
    }

    std::array<bool, kMaxRank> used{};
    for (size_t i = 0; i < axesAttr->GetSize(); ++i) {
        const int64_t rawAxis = axesData[i];
        const int64_t axis = rawAxis < 0 ? rawAxis + rank : rawAxis;
        if (axis < 0 || axis >= rank || used[static_cast<size_t>(axis)]) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(rawAxis).c_str(),
                                                  "each axis must be unique and within [-rank, rank)");
            return ge::GRAPH_FAILED;
        }
        used[static_cast<size_t>(axis)] = true;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShapeAndAxes(gert::InferShapeContext* context, const gert::Shape& xShape)
{
    if (Ops::Base::IsUnknownRank(xShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    if (rank <= 0 || rank > kMaxRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                              "rank must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }
    return ValidateAxes(context, rank);
}
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
    if (ValidateShapeAndAxes(context, *xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Centralization(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4Centralization.");
    if (ValidateDataType(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
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
