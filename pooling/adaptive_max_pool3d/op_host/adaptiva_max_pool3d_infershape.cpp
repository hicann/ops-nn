/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_max_pool3d_infershape.cpp
 * \brief
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;
using namespace std;

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t INDEX_OUTPUT_SIZE = 0;
constexpr size_t INDEX_OUT_MAX = 0;
constexpr size_t INDEX_OUT_INDICES = 1;
constexpr size_t INDEX_DTYPE = 1;
constexpr size_t INT32_DTYPE = 3;


constexpr size_t NCDHW_DIMS = 5;
constexpr size_t CDHW_DIMS = 4;
constexpr size_t OUTPUT_SIZE_DIMS = 3;
constexpr size_t ONE_DIMS = 1;
constexpr size_t NONE_DIMS = 0;

} // namespace

namespace ops {
static ge::graphStatus InferShape4AdaptiveMaxPool3d(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* indices_shape = context->GetOutputShape(INDEX_OUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    auto attr_ptr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_ptr);
    auto output_size_ptr = attr_ptr->GetAttrPointer<gert::ContinuousVector>(INDEX_OUTPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_size_ptr);
    auto output_size = static_cast<const int64_t*>(output_size_ptr->GetData());
    const int* index_dtype_ptr = attr_ptr->GetAttrPointer<int>(INDEX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, index_dtype_ptr);
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        Ops::Base::SetUnknownRank(*indices_shape);
        OP_LOGD(context->GetNodeName(), "AdaptiveMaxPool3d infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }
    size_t input_dim_num = x_shape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*x_shape)) {
        Ops::Base::SetUnknownShape(input_dim_num, *y_shape);
        Ops::Base::SetUnknownShape(input_dim_num, *indices_shape);
        OP_LOGD(context->GetNodeName(), "AdaptiveMaxPool3d infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }
    
    OP_CHECK_IF(
        input_dim_num != NCDHW_DIMS && input_dim_num != CDHW_DIMS,
        OP_LOGE(context, "The dims of x not equal 5 or 4."), return GRAPH_FAILED);
    y_shape->SetDimNum(input_dim_num);
    indices_shape->SetDimNum(input_dim_num);

    size_t output_size_len = output_size_ptr->GetSize();
    OP_CHECK_IF(
        output_size_len != OUTPUT_SIZE_DIMS && output_size_len != ONE_DIMS && output_size_len != NONE_DIMS,
        OP_LOGE(context, "The size of output_size is not equals to 0, 1 or 3."),
        return GRAPH_FAILED);
    
    for(size_t j = 0; j < input_dim_num - OUTPUT_SIZE_DIMS; j++) {
        y_shape->SetDim(j, x_shape->GetDim(j));
        indices_shape->SetDim(j, x_shape->GetDim(j));
    }

    std::vector<int> realOutDims = {};
    if (output_size_len  == OUTPUT_SIZE_DIMS) {
        for (size_t i = 0; i < OUTPUT_SIZE_DIMS; i++) {
            realOutDims.push_back(output_size[i]);
        }
    } else if (output_size_len  == ONE_DIMS) {
        for (size_t i = 0; i < OUTPUT_SIZE_DIMS; i++) {
            realOutDims.push_back(output_size[0]);
        }
    } else {
        for (size_t i = 0; i < OUTPUT_SIZE_DIMS; i++) {
            realOutDims.push_back(x_shape->GetDim(i + input_dim_num - OUTPUT_SIZE_DIMS));
        }
    }

    for (size_t i = 0; i < OUTPUT_SIZE_DIMS; i++) {
        y_shape->SetDim(i + input_dim_num - OUTPUT_SIZE_DIMS, realOutDims[i]);
        indices_shape->SetDim(i + input_dim_num - OUTPUT_SIZE_DIMS, realOutDims[i]);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4AdaptiveMaxPool3d(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType x = context->GetInputDataType(0);
    context->SetOutputDataType(INDEX_OUT_MAX, x);

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType indicesDtype = *dstDtype == INT32_DTYPE ? ge::DT_INT32 : ge::DT_INT64;

    context->SetOutputDataType(INDEX_OUT_INDICES, indicesDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdaptiveMaxPool3d)
    .InferShape(InferShape4AdaptiveMaxPool3d)
    .InferDataType(InferDtype4AdaptiveMaxPool3d);

} // namespace ops