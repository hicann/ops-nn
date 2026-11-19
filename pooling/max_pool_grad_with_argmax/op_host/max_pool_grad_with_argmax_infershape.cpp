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
    OP_CHECK_IF(output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
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
    const char* opName_ = "MaxPoolGradWithArgmax";

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolGradWithArgmax infershape running");
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto xOriFormat = xDesc->GetOriginFormat();
    if (xOriFormat != FORMAT_ND && xOriFormat != FORMAT_NCHW && xOriFormat != FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x", ge::TypeUtils::FormatToSerialString(xOriFormat).c_str(),
                                                "format only supports ND, NCHW, NHWC");
        return GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_KSIZE);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_STRIDES);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "strides", std::to_string(strides->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    if (ksize_data[INDEX_ZERO] != KSIZE_STRIDES_VALUE || strides_data[INDEX_ZERO] != KSIZE_STRIDES_VALUE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "ksize[0], strides[0]",
            (std::to_string(ksize_data[INDEX_ZERO]) + ", " + std::to_string(strides_data[INDEX_ZERO])).c_str(),
            "Pooling ksize[0] and strides[0] must be 1");
        return GRAPH_FAILED;
    }
    auto padsPtr = attrs->GetAttrPointer<char>(ATTR_INDEX_PADS);
    OP_CHECK_IF(
        padsPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "padding",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    std::string padding(padsPtr);
    if (padding != "SAME" && padding != "VALID") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pads", padding.c_str(), "Pads attribute must be SAME or VALID");
        return GRAPH_FAILED;
    }

    auto include_batch_in_index = attrs->GetAttrPointer<bool>(ATTR_INDEX_INCLUDE_BATCH_IN_INDEX);
    OP_CHECK_IF(
        include_batch_in_index == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "include_batch_in_index",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    OP_LOGE_IF(dataFormatPtr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get dataFormat failed.");
    std::string dataFormatStr = dataFormatPtr == nullptr ? "NULL" : dataFormatPtr;
    if (dataFormatStr == "NHWC") {
        if (ksize_data[INDEX_THREE] != KSIZE_STRIDES_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(opName_, "ksize[3]", std::to_string(ksize_data[INDEX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[INDEX_THREE] != KSIZE_STRIDES_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(opName_, "strides[3]", std::to_string(strides_data[INDEX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    } else {
        if (ksize_data[INDEX_ONE] != KSIZE_STRIDES_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(opName_, "ksize[1]", std::to_string(ksize_data[INDEX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[INDEX_ONE] != KSIZE_STRIDES_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(opName_, "strides[1]", std::to_string(strides_data[INDEX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const gert::Shape* gradShape = context->GetInputShape(1);
    OP_CHECK_IF(
        gradShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "grad",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const gert::Shape* argmaxShape = context->GetInputShape(2);
    OP_CHECK_IF(
        argmaxShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "argmax",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "yShape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*gradShape) ||
        Ops::Base::IsUnknownShape(*argmaxShape)) {
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

IMPL_OP_INFERSHAPE(MaxPoolGradWithArgmax).InferShape(InferShapeForMaxPoolGradWithArgmax);
} // namespace ops
