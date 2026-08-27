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
 * \file max_pool_v3_grad_infershape.cpp
 * \brief Shape inference implementation for MaxPoolV3Grad.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "util/shape_util.h"

namespace ops {

constexpr size_t ORIG_INPUT_INDEX = 0;
constexpr size_t ORIG_OUTPUT_INDEX = 1;
constexpr size_t GRAD_INDEX = 2;
constexpr size_t OUT_GRAD_INDEX = 0;

constexpr size_t ATTR_INDEX_KSIZE = 0;
constexpr size_t ATTR_INDEX_STRIDES = 1;
constexpr size_t ATTR_INDEX_PADDING_MODE = 2;
constexpr size_t ATTR_INDEX_PADS = 3;
constexpr size_t ATTR_INDEX_DATA_FORMAT = 4;
constexpr size_t ATTR_INDEX_GLOBAL_POOLING = 5;
constexpr size_t ATTR_INDEX_CEIL_MODE = 6;

constexpr size_t EXPECTED_RANK = 4;
constexpr size_t ATTR_LIST_SIZE = 4;

static ge::graphStatus CheckAttrsValid(const gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksizeAttr = attrs->GetListInt(ATTR_INDEX_KSIZE);
    auto stridesAttr = attrs->GetListInt(ATTR_INDEX_STRIDES);
    auto padsAttr = attrs->GetListInt(ATTR_INDEX_PADS);

    OP_CHECK_NULL_WITH_CONTEXT(context, ksizeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsAttr);

    if (ksizeAttr->GetSize() != ATTR_LIST_SIZE || stridesAttr->GetSize() != ATTR_LIST_SIZE ||
        padsAttr->GetSize() != ATTR_LIST_SIZE) {
        OP_LOGE(context->GetNodeName(), "ksize, strides and pads must contain four elements.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* ksize = ksizeAttr->GetData();
    const int64_t* strides = stridesAttr->GetData();
    const int64_t* pads = padsAttr->GetData();

    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);

    const char* paddingMode = attrs->GetAttrPointer<char>(ATTR_INDEX_PADDING_MODE);
    const char* dataFormat = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    const bool* globalPooling = attrs->GetAttrPointer<bool>(ATTR_INDEX_GLOBAL_POOLING);
    const bool* ceilMode = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);

    OP_CHECK_NULL_WITH_CONTEXT(context, paddingMode);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataFormat);
    OP_CHECK_NULL_WITH_CONTEXT(context, globalPooling);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceilMode);

    const std::string paddingModeString(paddingMode);
    const std::string dataFormatString(dataFormat);

    if (dataFormatString != "NCHW" && dataFormatString != "NHWC") {
        OP_LOGE(context->GetNodeName(), "data_format must be NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    if (paddingModeString != "CALCULATED" && paddingModeString != "SAME" && paddingModeString != "VALID") {
        OP_LOGE(context->GetNodeName(), "padding_mode must be CALCULATED, SAME or VALID.");
        return ge::GRAPH_FAILED;
    }

    const size_t channelIndex = dataFormatString == "NCHW" ? 1 : 3;
    const size_t heightIndex = dataFormatString == "NCHW" ? 2 : 1;
    const size_t widthIndex = dataFormatString == "NCHW" ? 3 : 2;

    if (strides[0] != 1 || strides[channelIndex] != 1) {
        OP_LOGE(context->GetNodeName(), "The N/C dimensions of strides must be 1.");
        return ge::GRAPH_FAILED;
    }

    if (strides[heightIndex] < 1 || strides[heightIndex] > 63 || strides[widthIndex] < 1 || strides[widthIndex] > 63) {
        OP_LOGE(context->GetNodeName(), "The H/W dimensions of strides must be in [1, 63].");
        return ge::GRAPH_FAILED;
    }

    if (!(*globalPooling)) {
        if (ksize[0] != 1 || ksize[channelIndex] != 1) {
            OP_LOGE(context->GetNodeName(), "The N/C dimensions of ksize must be 1.");
            return ge::GRAPH_FAILED;
        }

        if (ksize[heightIndex] < 1 || ksize[heightIndex] > 255 || ksize[widthIndex] < 1 || ksize[widthIndex] > 255) {
            OP_LOGE(context->GetNodeName(), "The H/W dimensions of ksize must be in [1, 255].");
            return ge::GRAPH_FAILED;
        }

        if (paddingModeString == "CALCULATED" && (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0)) {
            OP_LOGE(context->GetNodeName(), "pads must be nonnegative under CALCULATED mode.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleShape(const gert::InferShapeContext* context, const gert::Shape* origInputShape,
                                   const gert::Shape* origOutputShape, const gert::Shape* gradShape,
                                   gert::Shape* outGradShape)
{
    if (Ops::Base::IsUnknownRank(*origInputShape) || Ops::Base::IsUnknownRank(*origOutputShape) ||
        Ops::Base::IsUnknownRank(*gradShape)) {
        Ops::Base::SetUnknownRank(*outGradShape);
        OP_LOGD(context->GetNodeName(), "MaxPoolV3Grad infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownShape(*origInputShape) || Ops::Base::IsUnknownShape(*origOutputShape) ||
        Ops::Base::IsUnknownShape(*gradShape)) {
        Ops::Base::SetUnknownShape(EXPECTED_RANK, *outGradShape);
        OP_LOGD(context->GetNodeName(), "MaxPoolV3Grad infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (origInputShape->GetDimNum() != EXPECTED_RANK || origOutputShape->GetDimNum() != EXPECTED_RANK ||
        gradShape->GetDimNum() != EXPECTED_RANK) {
        OP_LOGE(context->GetNodeName(), "orig_input, orig_output and grad must be 4D tensors.");
        return ge::GRAPH_FAILED;
    }

    *outGradShape = *origInputShape;

    OP_LOGD(context->GetNodeName(), "MaxPoolV3Grad infershape run success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMaxPoolV3Grad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolV3Grad infershape running");

    auto origInputDesc = context->GetInputDesc(ORIG_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, origInputDesc);
    const ge::Format origInputFormat = origInputDesc->GetOriginFormat();
    if (origInputFormat != ge::FORMAT_ND && origInputFormat != ge::FORMAT_NCHW && origInputFormat != ge::FORMAT_NHWC) {
        OP_LOGE(context->GetNodeName(), "orig_input format must be ND, NCHW or NHWC.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = CheckAttrsValid(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape* origInputShape = context->GetInputShape(ORIG_INPUT_INDEX);
    const gert::Shape* origOutputShape = context->GetInputShape(ORIG_OUTPUT_INDEX);
    const gert::Shape* gradShape = context->GetInputShape(GRAD_INDEX);
    gert::Shape* outGradShape = context->GetOutputShape(OUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, origInputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, origOutputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outGradShape);

    return HandleShape(context, origInputShape, origOutputShape, gradShape, outGradShape);
}

IMPL_OP_INFERSHAPE(MaxPoolV3Grad).InferShape(InferShapeMaxPoolV3Grad);

} // namespace ops
