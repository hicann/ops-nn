/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kInputX = 0;
constexpr size_t kOutputY = 0;
constexpr size_t kAttrKsize = 0;
constexpr size_t kAttrStrides = 1;
constexpr size_t kAttrPads = 2;
constexpr size_t kAttrCeilMode = 3;
constexpr size_t kNchwWidth = 3;
constexpr size_t kNhwcWidth = 2;
constexpr size_t kOutputRank = 4;
constexpr size_t kPadsMinSize = 2;
constexpr int64_t kUnknownDim = -1;

template <typename Context>
ge::graphStatus CalculateOutputWidth(Context* context, int64_t inputWidth, int64_t& outputWidth)
{
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr ksize failed");
        return ge::GRAPH_FAILED;
    }
    const auto* ksize = attrs->template GetAttrPointer<int64_t>(kAttrKsize);
    if (ksize == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr ksize failed");
        return ge::GRAPH_FAILED;
    }
    const auto* strides = attrs->template GetAttrPointer<int64_t>(kAttrStrides);
    if (strides == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr strides failed");
        return ge::GRAPH_FAILED;
    }
    if (*strides == 0) {
        OP_LOGE(context->GetNodeName(), "Value of strides should not be 0");
        return ge::GRAPH_FAILED;
    }
    const auto* pads = attrs->template GetAttrPointer<gert::ContinuousVector>(kAttrPads);
    if (pads == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr pads_list failed!");
        return ge::GRAPH_FAILED;
    }
    const auto* ceilMode = attrs->template GetAttrPointer<bool>(kAttrCeilMode);
    if (ceilMode == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr ceil_mode failed");
        return ge::GRAPH_FAILED;
    }
    if (pads->GetSize() < kPadsMinSize) {
        OP_LOGE(context->GetNodeName(), "Size of pads_list must greater than 1!");
        return ge::GRAPH_FAILED;
    }
    const auto* padsData = static_cast<const int64_t*>(pads->GetData());
    if (padsData == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetOpAttr pads_list failed!");
        return ge::GRAPH_FAILED;
    }

    const int64_t padLeft = padsData[0];
    const int64_t padRight = padsData[1];
    if (*ceilMode) {
        outputWidth = (inputWidth + padLeft + padRight - *ksize + *strides - 1) / *strides + 1;
    } else {
        outputWidth = (inputWidth + padLeft + padRight - *ksize) / *strides + 1;
    }
    if (padLeft != 0 && (outputWidth - 1) * *strides >= inputWidth + padLeft) {
        --outputWidth;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWidthIndex(const gert::CompileTimeTensorDesc* inputDesc, const char* opName, size_t& widthIndex)
{
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto format = static_cast<ge::Format>(ge::GetPrimaryFormat(inputDesc->GetOriginFormat()));
    if (format == ge::FORMAT_NCHW) {
        widthIndex = kNchwWidth;
        return ge::GRAPH_SUCCESS;
    }
    if (format == ge::FORMAT_NHWC) {
        widthIndex = kNhwcWidth;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(opName, "Input format only support NCHW or NHWC");
    return ge::GRAPH_FAILED;
}

void SetOutputShape(ge::Format format, int64_t outputWidth, gert::Shape& outputShape)
{
    outputShape.SetDimNum(kOutputRank);
    if (format == ge::FORMAT_NCHW) {
        outputShape.SetDim(0, 1);
        outputShape.SetDim(1, 16);
        outputShape.SetDim(2, 1);
        outputShape.SetDim(3, outputWidth);
    } else {
        outputShape.SetDim(0, 1);
        outputShape.SetDim(1, 1);
        outputShape.SetDim(2, outputWidth);
        outputShape.SetDim(3, 16);
    }
}

int64_t GetDimWithLegacyDefault(const gert::Shape& shape, size_t index)
{
    return shape.GetDimNum() > index ? shape.GetDim(index) : 0;
}
} // namespace

static ge::graphStatus InferShapeForAvgPool1DAvgMatrix(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputShape = context->GetInputShape(kInputX);
    auto* outputShape = context->GetOutputShape(kOutputY);
    const auto* inputDesc = context->GetInputDesc(kInputX);
    if (inputShape == nullptr || outputShape == nullptr || inputDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get input shape, output shape or input desc failed");
        return ge::GRAPH_FAILED;
    }

    size_t widthIndex = 0;
    if (GetWidthIndex(inputDesc, context->GetNodeName(), widthIndex) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t outputWidth = GetDimWithLegacyDefault(*inputShape, widthIndex);
    if (outputWidth != kUnknownDim && CalculateOutputWidth(context, outputWidth, outputWidth) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetOutputShape(static_cast<ge::Format>(ge::GetPrimaryFormat(inputDesc->GetOriginFormat())), outputWidth,
                   *outputShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForAvgPool1DAvgMatrix(gert::InferShapeRangeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* inputRange = context->GetInputShapeRange(kInputX);
    auto* outputRange = context->GetOutputShapeRange(kOutputY);
    const auto* inputDesc = context->GetInputDesc(kInputX);
    if (inputRange == nullptr || inputRange->GetMin() == nullptr || inputRange->GetMax() == nullptr ||
        outputRange == nullptr || outputRange->GetMin() == nullptr || outputRange->GetMax() == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get input or output shape range failed");
        return ge::GRAPH_FAILED;
    }

    size_t widthIndex = 0;
    if (GetWidthIndex(inputDesc, context->GetNodeName(), widthIndex) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t minInputWidth = GetDimWithLegacyDefault(*inputRange->GetMin(), widthIndex);
    const int64_t maxInputWidth = GetDimWithLegacyDefault(*inputRange->GetMax(), widthIndex);
    int64_t minOutputWidth = 1;
    int64_t maxOutputWidth = kUnknownDim;
    if (minInputWidth == maxInputWidth && minInputWidth != kUnknownDim) {
        minOutputWidth = minInputWidth;
        if (CalculateOutputWidth(context, minInputWidth, minOutputWidth) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        maxOutputWidth = minOutputWidth;
    }

    const auto format = static_cast<ge::Format>(ge::GetPrimaryFormat(inputDesc->GetOriginFormat()));
    SetOutputShape(format, minOutputWidth, *outputRange->GetMin());
    SetOutputShape(format, maxOutputWidth, *outputRange->GetMax());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AvgPool1DAvgMatrix)
    .InferShape(InferShapeForAvgPool1DAvgMatrix)
    .InferShapeRange(InferShapeRangeForAvgPool1DAvgMatrix);
} // namespace ops
