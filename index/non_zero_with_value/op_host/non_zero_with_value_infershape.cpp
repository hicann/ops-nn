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
 * \file non_zero_with_value_infershape.cpp
 * \brief 静态 max-size 输出:value=[numel] / index=[2*numel] / count=[1](对齐 A2 NonZeroWithValueInfer)。
 */
#include <string>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr size_t INPUT_RANK_2D = 2;
constexpr int64_t INDEX_COORD_FACTOR = 2; // 每个非零 2 个坐标(行、列)
constexpr size_t OUT_VALUE = 0;
constexpr size_t OUT_INDEX = 1;
constexpr size_t OUT_COUNT = 2;
} // namespace

namespace ops {
graphStatus InferShape4NonZeroWithValue(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* valueShape = context->GetOutputShape(OUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    gert::Shape* indexShape = context->GetOutputShape(OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexShape);
    gert::Shape* countShape = context->GetOutputShape(OUT_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, countShape);

    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*valueShape);
        SetUnknownRank(*indexShape);
        SetUnknownRank(*countShape);
        return ge::GRAPH_SUCCESS;
    }

    // 严格 2D(对齐 A2 实机)
    OP_CHECK_IF(xShape->GetDimNum() != INPUT_RANK_2D,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(xShape->GetDimNum()).c_str(),
                                                         "NonZeroWithValue requires 2D input"),
                return ge::GRAPH_FAILED);

    const int64_t numel = xShape->GetDim(0) * xShape->GetDim(1);

    valueShape->SetDimNum(1);
    valueShape->SetDim(0, numel); // value = [numel]

    indexShape->SetDimNum(1);
    indexShape->SetDim(0, INDEX_COORD_FACTOR * numel); // index = [2*numel] 坐标主序展平

    countShape->SetDimNum(1);
    countShape->SetDim(0, 1); // count = [1]
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4NonZeroWithValue(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // value 同 x dtype(float);index 由 attr dtype(索引 1,默认 INT32)定;count 恒 INT32。
    context->SetOutputDataType(OUT_VALUE, context->GetInputDataType(0));
    const int64_t* dstDtype = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    context->SetOutputDataType(OUT_INDEX, static_cast<ge::DataType>(*dstDtype));
    context->SetOutputDataType(OUT_COUNT, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(NonZeroWithValue).InferShape(InferShape4NonZeroWithValue).InferDataType(InferDataType4NonZeroWithValue);
} // namespace ops
