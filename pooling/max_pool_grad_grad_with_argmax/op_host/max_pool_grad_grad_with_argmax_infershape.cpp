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
 * \file max_pool_grad_grad_with_argmax_infershape.cpp
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

static constexpr size_t INDEX_X = 0;
static constexpr size_t INDEX_GRAD = 1;
static constexpr size_t INDEX_ARGMAX = 2;
static constexpr size_t INDEX_OUTPUT = 0;

static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_THREE = 3;

static constexpr size_t EXPECTED_RANK = 4;
static constexpr int64_t KSIZE_STRIDES_VALUE = 1;

ge::graphStatus InferShapeForMaxPoolGradGradWithArgmax(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const char* opName = "MaxPoolGradGradWithArgmax";

    auto xDesc = context->GetInputDesc(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    const ge::Format xOriFormat = xDesc->GetOriginFormat();
    if (xOriFormat != ge::FORMAT_ND && xOriFormat != ge::FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "x", ge::TypeUtils::FormatToSerialString(xOriFormat).c_str(),
                                                "format only supports ND, NHWC");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "strides", std::to_string(strides->GetSize()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    auto ksizeData = static_cast<const int64_t*>(ksize->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, ksizeData);

    auto stridesData = static_cast<const int64_t*>(strides->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesData);

    if (ksizeData[INDEX_ZERO] != KSIZE_STRIDES_VALUE || stridesData[INDEX_ZERO] != KSIZE_STRIDES_VALUE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "ksize[0], strides[0]",
            (std::to_string(ksizeData[INDEX_ZERO]) + ", " + std::to_string(stridesData[INDEX_ZERO])).c_str(),
            "Pooling ksize[0] and strides[0] must be 1");
        return ge::GRAPH_FAILED;
    }

    auto padsPtr = attrs->GetAttrPointer<char>(ATTR_INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);

    const std::string padding(padsPtr);
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pads", padding.c_str(), "Pads attribute must be SAME or VALID");
        return ge::GRAPH_FAILED;
    }

    if (ksizeData[INDEX_THREE] != KSIZE_STRIDES_VALUE || stridesData[INDEX_THREE] != KSIZE_STRIDES_VALUE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "ksize[3], strides[3]",
            (std::to_string(ksizeData[INDEX_THREE]) + ", " + std::to_string(stridesData[INDEX_THREE])).c_str(),
            "Pooling ksize[3] and strides[3] must be 1");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* gradShape = context->GetInputShape(INDEX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);

    const gert::Shape* argmaxShape = context->GetInputShape(INDEX_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, argmaxShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*argmaxShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    const size_t argmaxDimNum = argmaxShape->GetDimNum();
    if (argmaxDimNum != EXPECTED_RANK) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "argmax rank", std::to_string(argmaxDimNum).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*xShape) || Ops::Base::IsUnknownRank(*gradShape)) {
        *yShape = *argmaxShape;
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*gradShape) ||
        Ops::Base::IsUnknownShape(*argmaxShape)) {
        *yShape = *argmaxShape;
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *argmaxShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPoolGradGradWithArgmax(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(INDEX_X);
    context->SetOutputDataType(INDEX_OUTPUT, xDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolGradGradWithArgmax)
    .InferShape(InferShapeForMaxPoolGradGradWithArgmax)
    .InferDataType(InferDataTypeForMaxPoolGradGradWithArgmax);

} // namespace ops
