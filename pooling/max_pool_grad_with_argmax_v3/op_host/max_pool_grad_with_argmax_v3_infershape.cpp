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
 * \file max_pool_grad_with_argmax_v3_infershape.cpp
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
static constexpr size_t ATTR_INDEX_KSIZE = 0;
static constexpr size_t ATTR_INDEX_STRIDES = 1;
static constexpr size_t ATTR_INDEX_PADS = 2;
static constexpr size_t ATTR_INDEX_DILATION = 4;
static constexpr size_t ATTR_INDEX_CEIL_MODE = 5;
static constexpr size_t ATTR_INDEX_DATA_FORMAT = 6;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 2;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPoolGradWithArgmaxV3(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmaxV3 infershape running");
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xOriFormat = xDesc->GetOriginFormat();
    OP_CHECK_IF(
        xOriFormat != FORMAT_ND && xOriFormat != FORMAT_NCHW && xOriFormat != FORMAT_NHWC,
        OP_LOGE(context->GetNodeName(), "format only supports ND, NCHW, NHWC"), return GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %lu must be 2!", ksize->GetSize()), return GRAPH_FAILED);

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %lu must be 2!", strides->GetSize()), return GRAPH_FAILED);

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        pads->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of pads %lu must be 2!", pads->GetSize()), return GRAPH_FAILED);

    auto dilation = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_DILATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilation);
    OP_CHECK_IF(
        dilation->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of dilation %lu must be 2!", dilation->GetSize()), return GRAPH_FAILED);

    auto ceil_mode = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);

    const char* data_format = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT); // todo 是否能匹配上
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* gradShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* argmaxShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, argmaxShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    size_t xDimNum = xShape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*gradShape) || Ops::Base::IsUnknownShape(*argmaxShape)) {
        SetAllUnknownDim(xDimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmaxV3 infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return GRAPH_SUCCESS;
    }
    yShape->SetDimNum(xDimNum);
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmaxV3 infershape run success.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPoolGradWithArgmaxV3(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolGradWithArgmaxV3)
    .InferShape(InferShapeForMaxPoolGradWithArgmaxV3)
    .InferDataType(InferDataTypeForMaxPoolGradWithArgmaxV3);
} // namespace ops
