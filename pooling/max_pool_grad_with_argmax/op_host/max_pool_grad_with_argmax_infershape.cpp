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
 * \file max_pool_grad_with_argmax_infershape.cpp
 * \brief
 */
#include <string>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t ATTR_INDEX_KSIZE = 0;
static constexpr size_t ATTR_INDEX_STRIDES = 1;
static constexpr size_t ATTR_INDEX_PADS = 2;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 4;
static constexpr size_t ATTR_INDEX_INCLUDE_BATCH_IN_INDEX = 3;
static constexpr size_t ATTR_INDEX_DATA_FORMAT = 4;
static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_ONE = 1;
static constexpr size_t INDEX_THREE = 3;
static constexpr size_t KSIZE_STRIDES_VALUE = 1;
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

ge::graphStatus InferShapeForMaxPoolGradWithArgmax(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmax infershape running");
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
        OP_LOGE(context->GetNodeName(), "Length of ksize %lu must be 4!", ksize->GetSize()), return GRAPH_FAILED);
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %lu must be 4!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    
    OP_CHECK_IF(
        ksize_data[INDEX_ZERO] != KSIZE_STRIDES_VALUE || strides_data[INDEX_ZERO] != KSIZE_STRIDES_VALUE,
        OP_LOGE(context->GetNodeName(), "Pooling ksize[0] / strides[0] must be 1."), return GRAPH_FAILED);

    auto padsPtr = attrs->GetAttrPointer<char>(ATTR_INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    std::string padding(padsPtr);
    OP_CHECK_IF(padding != "SAME" && padding != "VALID",
             OP_LOGE(context->GetNodeName(), "Pads attritube must be 'SAME' or 'VALID'!"),
             return GRAPH_FAILED);
    auto include_batch_in_index = attrs->GetAttrPointer<bool>(ATTR_INDEX_INCLUDE_BATCH_IN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, include_batch_in_index);

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    OP_LOGE_IF(dataFormatPtr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get dataFormat failed.");
    std::string dataFormatStr = dataFormatPtr == nullptr ? "NULL" : dataFormatPtr;
    if (dataFormatStr == "NHWC") {
        OP_CHECK_IF(
            ksize_data[INDEX_THREE] != KSIZE_STRIDES_VALUE,
            OP_LOGE(context->GetNodeName(), "Pooling ksize[3] must be 1."), return GRAPH_FAILED);
        OP_CHECK_IF(
            strides_data[INDEX_THREE] != KSIZE_STRIDES_VALUE,
            OP_LOGE(context->GetNodeName(), "Pooling strides[3] must be 1."), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            ksize_data[INDEX_ONE] != KSIZE_STRIDES_VALUE,
            OP_LOGE(context->GetNodeName(), "Pooling ksize[1] must be 1."), return GRAPH_FAILED);
        OP_CHECK_IF(
            strides_data[INDEX_ONE] != KSIZE_STRIDES_VALUE,
            OP_LOGE(context->GetNodeName(), "Pooling strides[1] must be 1."), return GRAPH_FAILED);
    }
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
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmax infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    yShape->SetDimNum(xDimNum);
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmax infershape run success.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPoolGradWithArgmax(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolGradWithArgmax)
    .InferShape(InferShapeForMaxPoolGradWithArgmax)
    .InferDataType(InferDataTypeForMaxPoolGradWithArgmax);
} // namespace ops