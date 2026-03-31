/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    int64_t output_size_num = output_size_ptr->GetSize();
    OP_CHECK_IF(
        output_size_num != LENS_TWO,
        OP_LOGE(context->GetNodeName(), "don't support output_size_dims not equal to 2 , infershape failed"),
        return ge::GRAPH_FAILED);

    int64_t in_dim = in_shape->GetDimNum();
    OP_CHECK_IF(
        in_dim != MIN_INPUT_DIMS && in_dim != MAX_INPUT_DIMS,
        OP_LOGE(context->GetNodeName(), "expect input tensor is 3D or 4D tensor , infershape failed"),
        return ge::GRAPH_FAILED);

    y_shape->SetDimNum(0);
    for (int i = 0; i < in_dim - output_size_num; i++) {
        y_shape->AppendDim(in_shape->GetDim(i));
    }
    const int64_t* output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    for (int i = 0; i < output_size_num; i++) {
        y_shape->AppendDim((int64_t)output_size[i]);
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