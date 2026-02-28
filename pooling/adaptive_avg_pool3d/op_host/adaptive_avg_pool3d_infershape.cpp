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
 * \file adaptive_avg_pool3d_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include <string>

using namespace ge;
using namespace std;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t OUTPUT_SIZE_INDEX = 0;
constexpr size_t DATA_FORTMAT_INDEX = 1;

constexpr size_t X_DIMS_4 = 4;
constexpr size_t X_DIMS_5 = 5;
constexpr size_t NUM_TWO = 2;
constexpr size_t NUM_ONE = 1;
constexpr size_t OUTPUT_SIZE_DIMS = 3;

} // namespace

namespace ops {
static ge::graphStatus InferShape4AdaptiveAvgPool3d(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    auto output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);
    auto output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }
    size_t input_dim_num = x_shape->GetDimNum();
    size_t output_size_len = output_size_ptr->GetSize();
    OP_CHECK_IF(input_dim_num != X_DIMS_4 && input_dim_num != X_DIMS_5,
            OP_LOGE(context, "The dims of x must be 4 or 5, but got %zu.", input_dim_num),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        output_size_len != OUTPUT_SIZE_DIMS,
        OP_LOGE(context, "The size of output_size not equal 3."),
        return GRAPH_FAILED);
    y_shape->SetDimNum(input_dim_num);

    const char* data_format = attr_ptr->GetAttrPointer<char>(DATA_FORTMAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);
    std::string data_format_str = data_format;
    size_t s_idx = 0; 
    if (data_format_str == "NCDHW") {
        // NCDHW --> 2,CDHW-->1
        s_idx = (input_dim_num == X_DIMS_5) ? NUM_TWO : NUM_ONE;
    } else { 
        // NDHWC -> 1, DHWC-->0
        s_idx = (input_dim_num == X_DIMS_5) ? NUM_ONE : 0;
    }
    int64_t target_d = output_size[0];
    int64_t target_h = output_size[NUM_ONE];
    int64_t target_w = output_size[NUM_TWO];
    for (size_t i = 0; i < input_dim_num; ++i) {
        if (i == s_idx) {
            y_shape->SetDim(i, target_d);
        } else if (i == s_idx + NUM_ONE) {
            y_shape->SetDim(i, target_h);
        } else if (i == s_idx + NUM_TWO) {
            y_shape->SetDim(i, target_w);
        } else {
            y_shape->SetDim(i, x_shape->GetDim(i));
        }
    }
    
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4AdaptiveAvgPool3d(gert::InferDataTypeContext* context)
{
    auto x_dtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, x_dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveAvgPool3d)
    .InferShape(InferShape4AdaptiveAvgPool3d)
    .InferDataType(InferDtype4AdaptiveAvgPool3d);

} // namespace ops