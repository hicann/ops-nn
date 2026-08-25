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
 * \file adaptive_avg_pool_2d.cpp
 * \brief
 */

#include "error_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
constexpr int LENS_TWO = 2;
constexpr int MIN_INPUT_DIMS = 3;
constexpr int MAX_INPUT_DIMS = 4;
static ge::graphStatus InferShape4AdaptiveAvgPool2d(gert::InferShapeContext* context)
{
    const char* opName_ = "AdaptiveAvgPool2d";
    OP_LOGD(context->GetNodeName(), "runtime2.0 AdaptiveAvgPool2d infershape running");
    const gert::Shape* in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    if (Ops::Base::IsUnknownRank(*in_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* output_size_ptr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);
    int64_t output_size_num = output_size_ptr->GetSize();
    if (output_size_num != LENS_TWO) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of output_size", std::to_string(output_size_num).c_str(), "2");
        return ge::GRAPH_FAILED;
    }

    int64_t in_dim = in_shape->GetDimNum();
    if (in_dim != MIN_INPUT_DIMS && in_dim != MAX_INPUT_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "x", std::to_string(in_dim).c_str(), "3 or 4");
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDimNum(0);
    for (int i = 0; i < in_dim - output_size_num; i++) {
        y_shape->AppendDim(in_shape->GetDim(i));
    }
    const int64_t* output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    for (int i = 0; i < output_size_num; i++) {
        y_shape->AppendDim(static_cast<int64_t>(output_size[i]));
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 AdaptiveAvgPool2d infershape run success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4AdaptiveAvgPool2d(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "AdaptiveAvgPool2dInferDtype enter");
    // Get input tout
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);

    OP_LOGD(context->GetNodeName(), "AdaptiveAvgPool2dInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool2d)
    .InferShape(InferShape4AdaptiveAvgPool2d)
    .InferDataType(InferDtype4AdaptiveAvgPool2d);
} // namespace ops
