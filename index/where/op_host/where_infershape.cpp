/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file where_infershape.cpp
 * \brief
 */

#include <string>
#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

namespace ops {

const int64_t kMaxDimSize = 8;

static ge::graphStatus ReportInferShapeFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {
        opName, "Operator shape inference execution failed. Preserve the execution logs and contact operator support."};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}

static ge::graphStatus InferShapeRangeWhere(gert::InferShapeRangeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferShapeRangeWhere running begin");
    auto x_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    OP_CHECK_IF(x_shape_range == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"),
                return ge::GRAPH_FAILED);
    if (output_shape_range == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "output shape range of y is null");
    }

    auto x_max_shape = x_shape_range->GetMax();
    auto x_shape_dimnum = x_max_shape->GetDimNum();
    int64_t output_shape_max = 1;
    size_t output_shape_dim_num = 2U;

    for (size_t i = 0U; i < x_shape_dimnum; i++) {
        if (x_max_shape->GetDim(i) < 0) {
            output_shape_max = -1;
            break;
        }
        output_shape_max *= x_max_shape->GetDim(i);
    }

    // set outputshape range
    output_shape_range->GetMax()->SetDimNum(output_shape_dim_num);
    output_shape_range->GetMax()->SetDim(0, output_shape_max);
    output_shape_range->GetMax()->SetDim(1, x_shape_dimnum);
    output_shape_range->GetMin()->SetDimNum(output_shape_dim_num);
    output_shape_range->GetMin()->SetDim(0, 0);
    output_shape_range->GetMin()->SetDim(1, x_shape_dimnum);

    OP_LOGI(context->GetNodeName(), "InferShapeRangeWhere running success");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Where(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin InferDataType4Where.");

    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Where(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferShape4Where running begin");

    auto inShape = context->GetInputShape(0);
    OP_CHECK_IF(inShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "output shape of y is null");
    }

    auto dims_num = inShape->GetDimNum();
    OP_CHECK_IF(dims_num < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(dims_num).c_str(),
                                                         "the rank of input x must be at least 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dims_num > kMaxDimSize,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(dims_num).c_str(),
                                                         "the rank of input x must be at most 8"),
                return ge::GRAPH_FAILED);

    outShape->SetDimNum(2U);
    if (Ops::Base::IsUnknownRank(*inShape)) {
        OP_LOGW("InferShape4Where", "the input shape is [-2], set output shape is [-1, -1]!");
        // input shape: unknown rank
        outShape->SetDim(0, ge::UNKNOWN_DIM);
        outShape->SetDim(1, ge::UNKNOWN_DIM);
    } else {
        // input shape: unknown dims
        // input shape: known
        outShape->SetDim(0, ge::UNKNOWN_DIM);
        outShape->SetDim(1, dims_num);
    }

    OP_LOGI(context->GetNodeName(), "InferShape4Where running success");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Where)
    .InferShape(InferShape4Where)
    .InferShapeRange(InferShapeRangeWhere)
    .InferDataType(InferDataType4Where);
} // namespace ops
