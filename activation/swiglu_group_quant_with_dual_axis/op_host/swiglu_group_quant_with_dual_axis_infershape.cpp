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
 * \file swiglu_group_quant_with_dual_axis_infershape.cpp
 * \brief Shape and dtype inference for SwigluGroupQuantWithDualAxis.
 */

#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "swiglu_group_quant_with_dual_axis_contract.h"

namespace ops {
namespace v2 = swiglu_group_quant_with_dual_axis;
namespace {
bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

int64_t ScalePairCount(int64_t extent)
{
    return extent < 0 ? v2::kUnknownDim : extent / v2::kScalePairElements + (extent % v2::kScalePairElements != 0);
}

void SetYShape(gert::Shape& shape, int64_t keep, int64_t h)
{
    shape.SetDimNum(0);
    shape.AppendDim(keep);
    shape.AppendDim(h);
}

void SetScale1Shape(gert::Shape& shape, int64_t keep, int64_t h)
{
    shape.SetDimNum(0);
    shape.AppendDim(keep);
    shape.AppendDim(ScalePairCount(h));
    shape.AppendDim(v2::kScalePair);
}

void SetEmpty(gert::Shape& shape)
{
    shape.SetDimNum(0);
    shape.AppendDim(0);
}
} // namespace

ge::graphStatus InferShape4SwigluGroupQuantWithDualAxis(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(v2::X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    gert::Shape* y1 = context->GetOutputShape(v2::Y1);
    gert::Shape* scale1 = context->GetOutputShape(v2::MX_SCALE1);
    gert::Shape* y2 = context->GetOutputShape(v2::Y2);
    gert::Shape* scale2 = context->GetOutputShape(v2::MX_SCALE2);
    gert::Shape* origin = context->GetOutputShape(v2::Y_ORIGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2);
    OP_CHECK_NULL_WITH_CONTEXT(context, origin);

    const int64_t* modeAttr = attrs->GetAttrPointer<int64_t>(v2::QUANT_MODE);
    const int64_t mode = modeAttr == nullptr ? v2::kDualAxisMode : *modeAttr;
    OP_CHECK_IF(mode != v2::kDualAxisMode,
                OP_LOGE(context->GetNodeName(), "attr quant_mode should be %ld, got %ld.", v2::kDualAxisMode, mode),
                return ge::GRAPH_FAILED);
    const bool* outputOriginAttr = attrs->GetAttrPointer<bool>(v2::OUTPUT_ORIGIN);
    const bool outputOrigin = outputOriginAttr != nullptr && *outputOriginAttr;

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*y1);
        Ops::Base::SetUnknownRank(*scale1);
        Ops::Base::SetUnknownRank(*y2);
        Ops::Base::SetUnknownRank(*scale2);
        if (outputOrigin) {
            Ops::Base::SetUnknownRank(*origin);
        } else {
            SetEmpty(*origin);
        }
        return ge::GRAPH_SUCCESS;
    }

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(rank != 2, OP_LOGE(context->GetNodeName(), "input x dim num should be 2, got %ld.", rank),
                return ge::GRAPH_FAILED);

    const gert::Shape* weightShape = context->GetOptionalInputShape(v2::WEIGHT);
    if (weightShape != nullptr) {
        const size_t weightRank = weightShape->GetDimNum();
        OP_CHECK_IF(weightRank < 1 || weightRank > 8,
                    OP_LOGE(context->GetNodeName(), "input weight dim num should be in [1, 8], got %zu.", weightRank),
                    return ge::GRAPH_FAILED);
    }

    int64_t t = 1;
    bool unknownT = false;
    for (int64_t i = 0; i + 1 < rank; ++i) {
        const int64_t extent = xShape->GetDim(i);
        if (extent == v2::kUnknownDim) {
            unknownT = true;
            continue;
        }
        OP_CHECK_IF(extent <= 0,
                    OP_LOGE(context->GetNodeName(), "input x dim[%ld] should be greater than 0, got %ld.", i, extent),
                    return ge::GRAPH_FAILED);
        int64_t next = 0;
        OP_CHECK_IF(
            !CheckedMul(t, extent, next),
            OP_LOGE(context->GetNodeName(),
                    "input x shape exceeds int64 range, got dim[%ld] %ld with accumulated product %ld.", i, extent, t),
            return ge::GRAPH_FAILED);
        t = next;
    }
    const int64_t cut = xShape->GetDim(rank - 1);
    OP_CHECK_IF(cut != v2::kUnknownDim && cut % v2::kScalePairElements != 0,
                OP_LOGE(context->GetNodeName(), "input x last dim should be divisible by 64, got %ld.", cut),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cut != v2::kUnknownDim && cut <= 0,
                OP_LOGE(context->GetNodeName(), "input x last dim should be greater than 0, got %ld.", cut),
                return ge::GRAPH_FAILED);
    const int64_t h = cut == v2::kUnknownDim ? v2::kUnknownDim : cut / 2;
    OP_CHECK_IF(
        h != v2::kUnknownDim && h < v2::kMxBlockSize,
        OP_LOGE(context->GetNodeName(), "input x last dim should be greater than or equal to 64, got %ld.", cut),
        return ge::GRAPH_FAILED);

    const gert::Shape* groupShape = context->GetOptionalInputShape(v2::GROUP_INDEX);
    int64_t groups = 1;
    const int64_t rows = unknownT ? v2::kUnknownDim : t;
    bool hasGroup = false;
    if (groupShape != nullptr) {
        OP_CHECK_IF(
            groupShape->GetDimNum() != 1,
            OP_LOGE(context->GetNodeName(), "input group_index dim num should be 1, got %zu.", groupShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        groups = groupShape->GetDim(0);
        OP_CHECK_IF(
            groups == 0 || groups < v2::kUnknownDim,
            OP_LOGE(context->GetNodeName(), "input group_index element num should be greater than 0, got %ld.", groups),
            return ge::GRAPH_FAILED);
        hasGroup = true;
    }

    if (!hasGroup) {
        *y1 = *xShape;
        y1->SetDim(rank - 1, h);
        *y2 = *y1;
        *scale1 = *y1;
        scale1->SetDim(rank - 1, ScalePairCount(h));
        scale1->AppendDim(v2::kScalePair);
        *scale2 = *y1;
        const int64_t batchRows = xShape->GetDim(rank - 2);
        scale2->SetDim(rank - 2, ScalePairCount(batchRows));
        scale2->AppendDim(v2::kScalePair);
        if (outputOrigin) {
            *origin = *y1;
        } else {
            SetEmpty(*origin);
        }
        return ge::GRAPH_SUCCESS;
    }

    SetYShape(*y1, rows, h);
    SetScale1Shape(*scale1, rows, h);
    SetYShape(*y2, rows, h);
    scale2->SetDimNum(0);
    int64_t scale2Rows = v2::kUnknownDim;
    if (rows >= 0 && groups >= 0) {
        const int64_t fullPairs = rows / v2::kScalePairElements;
        OP_CHECK_IF(groups > std::numeric_limits<int64_t>::max() - fullPairs,
                    OP_LOGE(context->GetNodeName(), "grouped mxscale2 row count exceeds int64 range."),
                    return ge::GRAPH_FAILED);
        scale2Rows = fullPairs + groups;
    }
    scale2->AppendDim(scale2Rows);
    scale2->AppendDim(h);
    scale2->AppendDim(v2::kScalePair);
    if (outputOrigin) {
        SetYShape(*origin, rows, h);
    } else {
        SetEmpty(*origin);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4SwigluGroupQuantWithDualAxis(gert::InferDataTypeContext* context)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dstTypeAttr = attrs->GetAttrPointer<int64_t>(v2::DST_TYPE);
    const int64_t dstTypeValue = dstTypeAttr == nullptr ? v2::kDstTypeE4M3Fn : *dstTypeAttr;
    OP_CHECK_IF(
        dstTypeValue != v2::kDstTypeE4M3Fn && dstTypeValue != v2::kDstTypeE5M2,
        OP_LOGE(context->GetNodeName(), "attr dst_type only supports %ld(FLOAT8_E5M2) or %ld(FLOAT8_E4M3FN), got %ld.",
                v2::kDstTypeE5M2, v2::kDstTypeE4M3Fn, dstTypeValue),
        return ge::GRAPH_FAILED);
    const auto dstType = dstTypeValue == v2::kDstTypeE5M2 ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;
    if (context->SetOutputDataType(v2::Y1, dstType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(v2::MX_SCALE1, ge::DT_FLOAT8_E8M0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(v2::Y2, dstType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(v2::MX_SCALE2, ge::DT_FLOAT8_E8M0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(v2::Y_ORIGIN, context->GetInputDataType(v2::X)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuantWithDualAxis)
    .InferShape(InferShape4SwigluGroupQuantWithDualAxis)
    .InferDataType(InferDtype4SwigluGroupQuantWithDualAxis);
} // namespace ops
