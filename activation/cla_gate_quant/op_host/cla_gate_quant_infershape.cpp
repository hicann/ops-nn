/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/shape_util.h"

namespace ops {
constexpr int64_t ROW_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t UNKNOWN_DIM = -1;
// Attr order must match op_def:
// dst_type(0), round_mode(1), scale_alg(2), input_attn_layout(3), dual_axis_flag(4)
constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t INDEX_ATTR_DUAL_AXIS_FLAG = 4;

ge::graphStatus InferShapeForClaGateQuant(gert::InferShapeContext* context)
{
    const gert::Shape* globalAttnShape = context->GetInputShape(0);
    gert::Shape* rowData = context->GetOutputShape(0);
    gert::Shape* rowScale = context->GetOutputShape(1);
    gert::Shape* colData = context->GetOutputShape(2);
    gert::Shape* colScale = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalAttnShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowScale);
    OP_CHECK_NULL_WITH_CONTEXT(context, colData);
    OP_CHECK_NULL_WITH_CONTEXT(context, colScale);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* dualAxisFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_DUAL_AXIS_FLAG);
    bool dualAxisFlag = dualAxisFlagPtr != nullptr ? *dualAxisFlagPtr : false;
    if (Ops::Base::IsUnknownRank(*globalAttnShape)) {
        Ops::Base::SetUnknownRank(*rowData);
        Ops::Base::SetUnknownRank(*rowScale);
        if (dualAxisFlag) {
            Ops::Base::SetUnknownRank(*colData);
            Ops::Base::SetUnknownRank(*colScale);
        } else {
            *colData = gert::Shape({0});
            *colScale = gert::Shape({0});
        }
        return ge::GRAPH_SUCCESS;
    }
    if (globalAttnShape->GetDimNum() != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "global_attn",
                                     std::to_string(globalAttnShape->GetDimNum()).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    int64_t t = globalAttnShape->GetDim(0);
    int64_t n = globalAttnShape->GetDim(1);
    int64_t d = globalAttnShape->GetDim(2);
    if (t == 0 || n == 0 || d == 0 || t < UNKNOWN_DIM || n < UNKNOWN_DIM || d < UNKNOWN_DIM) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "global_attn",
                                              Ops::Base::ToString(*globalAttnShape).c_str(),
                                              "each dimension must be positive or -1");
        return ge::GRAPH_FAILED;
    }
    int64_t k = (n == UNKNOWN_DIM || d == UNKNOWN_DIM) ? UNKNOWN_DIM : n * d;
    int64_t rowBlockNum = k == UNKNOWN_DIM ? UNKNOWN_DIM : Ops::Base::CeilDiv(k, ROW_BLOCK);
    int64_t colBlockNum = t == UNKNOWN_DIM ? UNKNOWN_DIM : Ops::Base::CeilDiv(t, ROW_BLOCK);
    *rowData = gert::Shape({t, k});
    *rowScale = gert::Shape({t, rowBlockNum, SCALE_LAST_DIM});
    if (!dualAxisFlag) {
        *colData = gert::Shape({0});
        *colScale = gert::Shape({0});
    } else {
        *colData = gert::Shape({t, k});
        *colScale = gert::Shape({colBlockNum, k, SCALE_LAST_DIM});
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForClaGateQuant(gert::InferDataTypeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t* dstType = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstType);
    if (outDtype != ge::DT_FLOAT4_E2M1 && outDtype != ge::DT_FLOAT4_E1M2 && outDtype != ge::DT_FLOAT8_E4M3FN &&
        outDtype != ge::DT_FLOAT8_E5M2) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "dst_type", Ops::Base::ToString(outDtype).c_str(),
                                  "FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN or FLOAT8_E5M2");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(2, outDtype);
    context->SetOutputDataType(3, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ClaGateQuant).InferShape(InferShapeForClaGateQuant).InferDataType(InferDataTypeForClaGateQuant);
} // namespace ops
