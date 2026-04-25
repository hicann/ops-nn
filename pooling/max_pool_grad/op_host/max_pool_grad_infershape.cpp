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
 * \file max_pool_grad_infershape.cpp
 * \brief
 */
#include <string>
#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

static constexpr size_t X1_INDEX = 0;
static constexpr size_t X2_INDEX = 1;
static constexpr size_t GRAD_INDEX = 2;
static constexpr size_t Y_INDEX = 0;

static constexpr size_t KSIZE_ATTR_INDEX = 0;
static constexpr size_t STRIDES_ATTR_INDEX = 1;
static constexpr size_t PADDING_ATTR_INDEX = 2;
static constexpr size_t DATA_FORMAT_ATTR_INDEX = 3;

static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static inline void SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
}

static ge::graphStatus CheckAttrsValid(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetListInt(KSIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != NCHW_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Length of ksize %lu must be 4!", ksize->GetSize()), return GRAPH_FAILED);

    auto strides = attrs->GetListInt(STRIDES_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != NCHW_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Length of strides %lu must be 4!", strides->GetSize()), return GRAPH_FAILED);

    const char* padding = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding);
    std::string paddingStr = padding;
    OP_CHECK_IF(
        paddingStr != "VALID" && paddingStr != "SAME",
        OP_LOGE(context->GetNodeName(), "padding only supports VALID or SAME, got %s", padding), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus HandleUnknownShape(
    gert::InferShapeContext* context, const gert::Shape* x1Shape, const gert::Shape* x2Shape,
    const gert::Shape* gradShape, gert::Shape* yShape)
{
    size_t x1DimNum = x1Shape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*x1Shape) || Ops::Base::IsUnknownShape(*x2Shape) ||
        Ops::Base::IsUnknownShape(*gradShape)) {
        SetAllUnknownDim(x1DimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*x1Shape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }
    yShape->SetDimNum(x1DimNum);
    *yShape = *x1Shape;
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape run success.");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPoolGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGrad infershape running");

    auto x1Desc = context->GetInputDesc(X1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto xOriFormat = x1Desc->GetOriginFormat();
    OP_CHECK_IF(
        xOriFormat != FORMAT_ND && xOriFormat != FORMAT_NCHW && xOriFormat != FORMAT_NHWC,
        OP_LOGE(context->GetNodeName(), "format only supports ND, NCHW, NHWC"), return GRAPH_FAILED);

    ge::graphStatus ret = CheckAttrsValid(context);
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape* x1Shape = context->GetInputShape(X1_INDEX);
    const gert::Shape* x2Shape = context->GetInputShape(X2_INDEX);
    const gert::Shape* gradShape = context->GetInputShape(GRAD_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    return HandleUnknownShape(context, x1Shape, x2Shape, gradShape, yShape);
}

static ge::graphStatus InferDataTypeForMaxPoolGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(X1_INDEX);
    context->SetOutputDataType(Y_INDEX, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolGrad).InferShape(InferShapeForMaxPoolGrad).InferDataType(InferDataTypeForMaxPoolGrad);
} // namespace ops
