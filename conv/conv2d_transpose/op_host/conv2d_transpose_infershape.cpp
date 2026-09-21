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
 * \file conv2d_transpose_infershape.cpp
 * \brief InferShape registration for legacy Conv2DTranspose, which is converted to
 *        Conv3DTransposeV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include <vector>
#include "common/op_host/conv_backprop_infershape.h"
#include "common/op_host/conv_common_cube_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kInputSizeIndex = 0;
constexpr size_t kXIndex = 1;
constexpr size_t kFilterIndex = 2;

constexpr size_t kStridesIdx = 0;
constexpr size_t kPadsIdx = 1;
constexpr size_t kDilationsIdx = 2;
constexpr size_t kGroupsIdx = 3;
constexpr size_t kOutputPaddingIdx = 5;
constexpr size_t kOutputShapeIdx = 9;
constexpr size_t kConv2dAttrListLen = 4;
constexpr size_t kPadTopIdx = 0;
constexpr size_t kPadBottomIdx = 1;
constexpr size_t kPadLeftIdx = 2;
constexpr size_t kPadRightIdx = 3;

struct Conv2DFormatPos {
    size_t n;
    size_t c;
    size_t h;
    size_t w;
};

static bool GetConv2DFormatPos(const ge::Format& format, Conv2DFormatPos& pos)
{
    if (format == ge::FORMAT_NCHW) {
        pos = {0, 1, 2, 3};
    } else if (format == ge::FORMAT_NHWC) {
        pos = {0, 3, 1, 2};
    } else if (format == ge::FORMAT_HWCN) {
        pos = {3, 2, 0, 1};
    } else {
        return false;
    }
    return true;
}

// input_size为const且x为静态shape时，filter的H/W维度不支持-1/-2
static ge::graphStatus CheckFilterShapeForConv2DTranspose(gert::InferShapeContext* context)
{
    const auto inputSizeTensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(inputSizeTensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input_size tensor."),
                return ge::GRAPH_FAILED);
    const auto xShape = context->GetInputShape(kXIndex);
    OP_CHECK_IF(xShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get x shape."),
                return ge::GRAPH_FAILED);
    const auto& inputSizeShape = inputSizeTensor->GetOriginShape();
    if (!IsConstSizeTensor(inputSizeTensor) || Ops::Base::IsUnknownRank(inputSizeShape) ||
        Ops::Base::IsUnknownShape(inputSizeShape) || !IsStaticShape(*xShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const auto filterShape = context->GetInputShape(kFilterIndex);
    OP_CHECK_IF(filterShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape."),
                return ge::GRAPH_FAILED);
    const auto filterDesc = context->GetInputDesc(kFilterIndex);
    OP_CHECK_IF(filterDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter tensor desc."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*filterShape)) {
        OP_LOGE(context->GetNodeName(),
                "filter shape [-2] should be 4 dims and h, w should be positive when input_size is const and x shape "
                "is static.");
        return ge::GRAPH_FAILED;
    }
    size_t hIndex = 0;
    size_t wIndex = 0;
    if (filterShape->GetDimNum() != kConv2dDimSizeLimit ||
        !GetHwDimIndex(filterDesc->GetOriginFormat(), hIndex, wIndex)) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t filterH = filterShape->GetDim(hIndex);
    const int64_t filterW = filterShape->GetDim(wIndex);
    if (filterH < 0 || filterW < 0) {
        OP_LOGE(context->GetNodeName(),
                "filter shape h = %ld, w = %ld should be positive when input_size is const and x shape is static.",
                filterH, filterW);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// onnx插件(conv_transpose_plugin.cc)合成的input_size恒为{0,0,0,0}占位符，全0时不采信const值，
// 按ONNX语义重算输出shape：优先output_shape属性(已统一为HW布局)，否则按反卷积公式
// stride*(x-1)+output_padding+(filter-1)*dilation+1-(pad_top+pad_bottom)；
static ge::graphStatus FixZeroPlaceholderShape(gert::InferShapeContext* context)
{
    const auto yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get y shape."),
                return ge::GRAPH_FAILED);
    if (!CheckOutputAllZero(yShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const auto xShape = context->GetInputShape(kXIndex);
    const auto xDesc = context->GetInputDesc(kXIndex);
    const auto filterShape = context->GetInputShape(kFilterIndex);
    const auto filterDesc = context->GetInputDesc(kFilterIndex);
    OP_CHECK_IF(xShape == nullptr || xDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get x shape or desc."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(filterShape == nullptr || filterDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape or desc."),
                return ge::GRAPH_FAILED);
    Conv2DFormatPos xPos;
    Conv2DFormatPos filterPos;
    OP_CHECK_IF(!GetConv2DFormatPos(xDesc->GetOriginFormat(), xPos) ||
                    !GetConv2DFormatPos(filterDesc->GetOriginFormat(), filterPos),
                OP_LOGE(context->GetNodeName(), "x format %s and filter format %s only support NCHW, NHWC or HWCN.",
                        ge::TypeUtils::FormatToAscendString(xDesc->GetOriginFormat()).GetString(),
                        ge::TypeUtils::FormatToAscendString(filterDesc->GetOriginFormat()).GetString()),
                return ge::GRAPH_FAILED);
    const int64_t xN = xShape->GetDim(xPos.n);
    const int64_t xH = xShape->GetDim(xPos.h);
    const int64_t xW = xShape->GetDim(xPos.w);
    const int64_t filterC = filterShape->GetDim(filterPos.c);
    const int64_t filterH = filterShape->GetDim(filterPos.h);
    const int64_t filterW = filterShape->GetDim(filterPos.w);
    OP_CHECK_IF(xN <= 0 || xH <= 0 || xW <= 0 || filterC <= 0 || filterH <= 0 || filterW <= 0,
                OP_LOGE(context->GetNodeName(),
                        "x shape and filter shape should be positive when input_size is all-zero placeholder."),
                return ge::GRAPH_FAILED);

    const auto runtimeAttrs = context->GetAttrs();
    OP_CHECK_IF(runtimeAttrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs."),
                return ge::GRAPH_FAILED);
    const auto stridesList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kStridesIdx);
    const auto padsList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kPadsIdx);
    const auto dilationsList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kDilationsIdx);
    const auto outputPaddingList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kOutputPaddingIdx);
    const int64_t* groups = runtimeAttrs->GetAttrPointer<int64_t>(kGroupsIdx);
    OP_CHECK_IF(stridesList == nullptr || padsList == nullptr || dilationsList == nullptr ||
                    outputPaddingList == nullptr || groups == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get strides/pads/dilations/groups attrs."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stridesList->GetSize() != kConv2dAttrListLen || padsList->GetSize() != kConv2dAttrListLen ||
                    dilationsList->GetSize() != kConv2dAttrListLen ||
                    outputPaddingList->GetSize() != kConv2dAttrListLen,
                OP_LOGE(context->GetNodeName(),
                        "strides/pads/dilations/output_padding size should be 4 when input_size is all-zero "
                        "placeholder."),
                return ge::GRAPH_FAILED);
    const auto strides = static_cast<const int64_t*>(stridesList->GetData());
    const auto pads = static_cast<const int64_t*>(padsList->GetData());
    const auto dilations = static_cast<const int64_t*>(dilationsList->GetData());
    const auto outputPadding = static_cast<const int64_t*>(outputPaddingList->GetData());
    const int64_t strideH = strides[xPos.h];
    const int64_t strideW = strides[xPos.w];
    const int64_t dilationH = dilations[xPos.h];
    const int64_t dilationW = dilations[xPos.w];
    const int64_t outputPaddingH = outputPadding[xPos.h];
    const int64_t outputPaddingW = outputPadding[xPos.w];
    OP_CHECK_IF(strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 || *groups <= 0,
                OP_LOGE(context->GetNodeName(),
                        "strides, dilations and groups should be positive when input_size is all-zero placeholder."),
                return ge::GRAPH_FAILED);

    int64_t outputH = 0;
    int64_t outputW = 0;
    // output_shape属性已由onnx插件统一为HW(2d)布局，前两元素>0时优先采信
    if (runtimeAttrs->GetAttrNum() > kOutputShapeIdx) {
        const auto outputShapeList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kOutputShapeIdx);
        if (outputShapeList != nullptr && outputShapeList->GetSize() >= 2) {
            const auto outputShapeData = static_cast<const int64_t*>(outputShapeList->GetData());
            if (outputShapeData[0] > 0 && outputShapeData[1] > 0) {
                outputH = outputShapeData[0];
                outputW = outputShapeData[1];
            }
        }
    }
    if (outputH == 0 || outputW == 0) {
        outputH = strideH * (xH - 1) + (outputPaddingH + (filterH - 1) * dilationH + 1) -
                  (pads[kPadTopIdx] + pads[kPadBottomIdx]);
        outputW = strideW * (xW - 1) + (outputPaddingW + (filterW - 1) * dilationW + 1) -
                  (pads[kPadLeftIdx] + pads[kPadRightIdx]);
    }
    OP_CHECK_IF(outputH <= 0 || outputW <= 0,
                OP_LOGE(context->GetNodeName(),
                        "inferred output shape h = %ld, w = %ld should be positive when input_size is all-zero "
                        "placeholder.",
                        outputH, outputW),
                return ge::GRAPH_FAILED);

    yShape->SetDim(xPos.n, xN);
    yShape->SetDim(xPos.c, filterC * (*groups));
    yShape->SetDim(xPos.h, outputH);
    yShape->SetDim(xPos.w, outputW);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Conv2DTranspose(gert::InferShapeContext* context)
{
    auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
    if (ret == ge::GRAPH_SUCCESS) {
        ret = CheckFilterShapeForConv2DTranspose(context);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        ret = FixZeroPlaceholderShape(context);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        auto yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "[InferShape] Conv2DTranspose y_shape: %s",
                    Ops::Base::ToString(*yShape).c_str());
        }
    }
    return ret;
}

} // namespace Conv

IMPL_OP_INFERSHAPE(Conv2DTranspose)
    .InferShape(Ops::NN::Conv::InferShape4Conv2DTranspose)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvTransposeV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("auto_pad", "NOTSET")
    .PrivateAttr("output_shape", std::vector<int64_t>{})
    .PrivateAttr("_op_impl_mode_enum", -1L);

} // namespace NN
} // namespace Ops
