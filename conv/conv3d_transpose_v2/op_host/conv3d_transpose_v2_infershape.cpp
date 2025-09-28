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
 * \file conv3d_transpose_v2_infershape.cpp
 * \brief
 */

#include <algorithm>
#include <log/log.h>
#include <graph/utils/type_utils.h>
#include "error_util.h"
#include "conv/common/op_host/conv_backprop_infershape.h"
#include "conv/common/op_host/cube_util.h"

using namespace std;

namespace {
constexpr size_t kConv2dInputSizeLimit = 4;
constexpr size_t kDeconvolutionStridesLimit = 2;
constexpr size_t kConv3dInputSizeLimit = 5;
constexpr size_t kConv3dPadsSizeLimit = 6;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t DIM_5 = 5;
constexpr size_t LEN_1 = 1;
constexpr size_t LEN_3 = 3;
constexpr size_t LEN_6 = 6;

constexpr size_t kConv3DStridesIdx = 0;
constexpr size_t kConv3DDilationsIdx = 2;

constexpr size_t kConv2DStridesIdx = 0;
constexpr size_t kConv2DDilationsIdx = 2;

}  // namespace

namespace Ops {
namespace NN {
namespace Conv {

using ge::Format;
using ge::FORMAT_DHWCN;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NCHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NHWC;
using gert::InferShapeContext;

static bool GetConv3DXShape(
    const InferShapeContext* context, size_t x_idx, Format x_format, bool avg_pool3d, Conv3DInputShapes& shapes)
{
    const auto x_shape = context->GetInputShape(x_idx);
    OP_LOGE_IF(x_shape == nullptr, false, context->GetNodeName(), "failed to get x shape.");
    OP_LOGE_IF(
        x_shape->GetDimNum() != kConv3dInputSizeLimit, false, context->GetNodeName(),
        "The shape of x_tensor should be 5d, actual dim num: %zu", x_shape->GetDimNum());

    size_t idx = 0;
    if (x_format == FORMAT_NCDHW) {
        shapes.in = x_shape->GetDim(idx++);
        shapes.ic = x_shape->GetDim(idx++);
        shapes.id = x_shape->GetDim(idx++);
        shapes.ih = x_shape->GetDim(idx++);
        shapes.iw = x_shape->GetDim(idx++);
    } else if (x_format == FORMAT_NDHWC) {
        shapes.in = x_shape->GetDim(idx++);
        shapes.id = x_shape->GetDim(idx++);
        shapes.ih = x_shape->GetDim(idx++);
        shapes.iw = x_shape->GetDim(idx++);
        shapes.ic = x_shape->GetDim(idx++);
    } else if (avg_pool3d && x_format == FORMAT_DHWCN) {
        shapes.id = x_shape->GetDim(idx++);
        shapes.ih = x_shape->GetDim(idx++);
        shapes.iw = x_shape->GetDim(idx++);
        shapes.ic = x_shape->GetDim(idx++);
        shapes.in = x_shape->GetDim(idx++);
    } else {
        OP_LOGE(
            context->GetNodeName(), "The format of input x not support format %s.",
            ge::TypeUtils::FormatToAscendString(x_format).GetString());
        return false;
    }

    return true;
}

static bool GetConv3DFilterShape(const InferShapeContext* context, size_t filter_idx, Conv3DInputShapes& shapes)
{
    const auto filter_desc = context->GetInputDesc(filter_idx);
    OP_LOGE_IF(filter_desc == nullptr, false, context->GetNodeName(), "failed to get filter tensor desc.");
    const auto filter_format = filter_desc->GetOriginFormat();
    const auto filter_shape = context->GetInputShape(filter_idx);
    OP_LOGE_IF(filter_shape == nullptr, false, context->GetNodeName(), "failed to get filter shape.");
    // already checked in shape range infer logic, no need to use error manager here
    OP_LOGE_IF(
        filter_shape->GetDimNum() != kConv3dInputSizeLimit, false, context->GetNodeName(),
        "The shape of the filter should be 5d, actual dim num: %zu", filter_shape->GetDimNum());

    size_t idx = 0;
    if (filter_format == FORMAT_NCDHW) {
        shapes.kn = filter_shape->GetDim(idx++);
        shapes.kc = filter_shape->GetDim(idx++);
        shapes.kd = filter_shape->GetDim(idx++);
        shapes.kh = filter_shape->GetDim(idx++);
        shapes.kw = filter_shape->GetDim(idx++);
    } else if (filter_format == FORMAT_NDHWC) {
        shapes.kn = filter_shape->GetDim(idx++);
        shapes.kd = filter_shape->GetDim(idx++);
        shapes.kh = filter_shape->GetDim(idx++);
        shapes.kw = filter_shape->GetDim(idx++);
        shapes.kc = filter_shape->GetDim(idx++);
    } else if (filter_format == FORMAT_DHWCN) {
        shapes.kd = filter_shape->GetDim(idx++);
        shapes.kh = filter_shape->GetDim(idx++);
        shapes.kw = filter_shape->GetDim(idx++);
        shapes.kc = filter_shape->GetDim(idx++);
        shapes.kn = filter_shape->GetDim(idx++);
    } else {
        OP_LOGE(
            context->GetNodeName(), "The format of input filter not support format %s.",
            ge::TypeUtils::FormatToAscendString(filter_format).GetString());
        return false;
    }

    return true;
}

static bool GetConv3DStridesAndDilations(const InferShapeContext* context, Format x_format, Conv3DAttrs& attrs)
{
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
    const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DStridesIdx);
    OP_LOGE_IF(strides_list == nullptr, false, context->GetNodeName(), "failed to get strides attrs");
    // already checked in first infer shape logic, double check just for security
    OP_LOGE_IF(strides_list->GetSize() != kConv3dInputSizeLimit, false, context->GetNodeName(), "invalid strides");

    const auto dilations_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DDilationsIdx);
    OP_LOGE_IF(dilations_list == nullptr, false, context->GetNodeName(), "failed to get dilations attrs");
    OP_LOGE_IF(dilations_list->GetSize() != kConv3dInputSizeLimit, false, context->GetNodeName(), "invalid dilations");

    const auto strides = static_cast<const int64_t*>(strides_list->GetData());
    const auto dilations = static_cast<const int64_t*>(dilations_list->GetData());
    if (x_format == ge::FORMAT_NCDHW) {
        attrs.strd = strides[kDDimNCDHWIdx];
        attrs.strh = strides[kHDimNCDHWIdx];
        attrs.strw = strides[kWDimNCDHWIdx];
        attrs.dild = dilations[kDDimNCDHWIdx];
        attrs.dilh = dilations[kHDimNCDHWIdx];
        attrs.dilw = dilations[kWDimNCDHWIdx];
    } else {
        // FORMAT_NDHWC, already checked in GetConv3DXShape, else is enough
        attrs.strd = strides[kDDimNDHWCIdx];
        attrs.strh = strides[kHDimNDHWCIdx];
        attrs.strw = strides[kWDimNDHWCIdx];
        attrs.dild = dilations[kDDimNDHWCIdx];
        attrs.dilh = dilations[kHDimNDHWCIdx];
        attrs.dilw = dilations[kWDimNDHWCIdx];
    }

    OP_LOGE_IF(
        attrs.strd == 0 || attrs.strh == 0 || attrs.strw == 0, false, context->GetNodeName(), "get zero strides");
    return true;
}

static bool GetConv3DPads(
    const InferShapeContext* context, const Conv3DInputShapes& shapes, size_t pads_idx, size_t padding_idx,
    Conv3DAttrs& attrs)
{
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
    const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(pads_idx);
    OP_LOGE_IF(pads_list == nullptr, false, context->GetNodeName(), "failed to get pads attrs");
    OP_LOGE_IF(
        pads_list->GetSize() != LEN_6 and pads_list->GetSize() != LEN_3 and pads_list->GetSize() != LEN_1, false,
        context->GetNodeName(), "invalid pads");
    const auto pads_list_data = static_cast<const int64_t*>(pads_list->GetData());

    if (pads_list->GetSize() == LEN_1 || pads_list->GetSize() == LEN_3) {
        attrs.padf = pads_list_data[DIM_0];
        attrs.padb = pads_list_data[DIM_0];
        attrs.padu = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_1];
        attrs.padd = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_1];
        attrs.padl = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_2];
        attrs.padr = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_2];
    } else {
        attrs.padf = pads_list_data[DIM_0];
        attrs.padb = pads_list_data[DIM_1];
        attrs.padu = pads_list_data[DIM_2];
        attrs.padd = pads_list_data[DIM_3];
        attrs.padl = pads_list_data[DIM_4];
        attrs.padr = pads_list_data[DIM_5];
    }

    if (runtime_attrs->GetAttrNum() > padding_idx) {
        const auto padding = runtime_attrs->GetAttrPointer<char>(padding_idx);
        if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
            OP_LOGD(context->GetNodeName(), "get padding SAME");
            int64_t tails_d = shapes.id % attrs.strd; // non zero, checked in shape range infer logic
            int64_t tails_h = shapes.ih % attrs.strh; // non zero, checked in shape range infer logic
            int64_t tails_w = shapes.iw % attrs.strw; // non zero, checked in shape range infer logic
            int64_t dilate_kernel_d = attrs.dild * (shapes.kd - 1) + 1;
            int64_t dilate_kernel_h = attrs.dilh * (shapes.kh - 1) + 1;
            int64_t dilate_kernel_w = attrs.dilw * (shapes.kw - 1) + 1;
            int64_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - attrs.strd), 0L);
            int64_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.strh), 0L);
            int64_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.strw), 0L);
            attrs.padf = pad_d / 2; // 2 means pad_up is half size of pad_d
            attrs.padb = pad_d - attrs.padf;
            attrs.padu = pad_h / 2; // 2 means pad_up is half size of pad_h
            attrs.padd = pad_h - attrs.padu;
            attrs.padl = pad_w / 2; // 2 means pad_up is half size of pad_w
            attrs.padr = pad_w - attrs.padl;
            return true;
        }
    }

    bool negative_pad =
        std::any_of(pads_list_data, pads_list_data + pads_list->GetSize(), [](int64_t val) -> bool { return val < 0; });
    OP_LOGE_IF(negative_pad, false, context->GetNodeName(), "The value of the pads attribute should >= 0");

    return true;
}

static bool GetConvGroups(const gert::InferShapeContext* context, size_t groups_idx, Conv3DAttrs& attrs)
{
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
    const auto groups = runtime_attrs->GetAttrPointer<int64_t>(groups_idx);
    OP_LOGE_IF(groups == nullptr, false, context->GetNodeName(), "failed to get groups attrs");
    attrs.groups = *groups;
    return true;
}

constexpr size_t kConv3dDimSizeLimit = 5;
constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kConv3DTransposeFilterIdx = 2;
constexpr size_t kConv3DTransposePadsIdx = 1;
constexpr size_t kConv3DTransposeGroupsIdx = 3;
constexpr size_t kConv3DTransposeOutputPaddingIdx = 5;
constexpr size_t kConv3DTransposePaddingIdx = 7;

static bool GetConv3DTransposeOutputPadding(const gert::InferShapeContext* const context,
                                            ge::Format x_format, int64_t output_padding[3],
                                            size_t output_padding_length)
{
    constexpr size_t output_padding_length_limit = 3;
    OP_LOGE_IF(output_padding == nullptr, false, context->GetNodeName(), "utput_padding is nullptr");
    OP_LOGE_IF(output_padding_length < output_padding_length_limit, false, context->GetNodeName(),
              "output_padding_length is less 3");
    // 3: DHW
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
    const auto output_padding_list =
        runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kConv3DTransposeOutputPaddingIdx);
    OP_LOGE_IF(output_padding_list == nullptr, false, context->GetNodeName(), "failed to get output_padding attrs");
    OP_LOGE_IF(
        output_padding_list->GetSize() != kConv3dDimSizeLimit, false, context->GetNodeName(), "invalid output_padding");

    const auto output_padding_data = static_cast<const int64_t*>(output_padding_list->GetData());
    size_t idx = 0;
    if (x_format == ge::FORMAT_NCDHW) {
        output_padding[idx++] = output_padding_data[kDDimNCDHWIdx];
        output_padding[idx++] = output_padding_data[kHDimNCDHWIdx];
        output_padding[idx++] = output_padding_data[kWDimNCDHWIdx];
    } else {
        // FORMAT_NDHWC, already checked in GetConv3DXShape, else is enough
        output_padding[idx++] = output_padding_data[kDDimNDHWCIdx];
        output_padding[idx++] = output_padding_data[kHDimNDHWCIdx];
        output_padding[idx++] = output_padding_data[kWDimNDHWCIdx];
    }

    return true;
}

static ge::graphStatus InferShapeForConv3DTranspose(gert::InferShapeContext* context)
{
    auto const_tensor = context->GetInputTensor(0);
    OP_CHECK_IF(
        const_tensor == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get null tensor"),
        return ge::GRAPH_FAILED);
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
    auto ret = ge::GRAPH_SUCCESS;
    if (const_tensor_dim_num == kConv2dDimSizeLimit) {
        ret = InferShapeForConvBackpropExtend3D(context, 0, "input_size");
    } else {
        ret = InferShapeForConvBackprop(context, 0, "input_size", kConv3dDimSizeLimit);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(
        y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "y shape is null"), return ge::GRAPH_FAILED);

    if (CheckOutputAllZero(y_shape)) {
        const auto x_desc = context->GetInputDesc(0);
        OP_CHECK_IF(
            x_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x desc is null"),
            return ge::GRAPH_FAILED);
        const auto x_format = x_desc->GetOriginFormat();

        Conv3DInputShapes shapes;
        Conv3DAttrs attrs;
        int64_t output_padding[3]; // 3: DHW
        if (GetConv3DXShape(context, 1UL, x_format, false, shapes) &&
            GetConv3DFilterShape(context, kConv3DTransposeFilterIdx, shapes) &&
            GetConv3DStridesAndDilations(context, x_format, attrs) &&
            GetConvGroups(context, kConv3DTransposeGroupsIdx, attrs) &&
            GetConv3DPads(context, shapes, kConv3DTransposePadsIdx, kConv3DTransposePaddingIdx, attrs) &&
            GetConv3DTransposeOutputPadding(context, x_format, output_padding, sizeof(output_padding)/sizeof(int64_t))) {
            int64_t output_d = attrs.strd * (shapes.id - 1) + output_padding[0] + ((shapes.kd - 1) * attrs.dild + 1) -
                               (attrs.padf + attrs.padb);
            int64_t output_h = attrs.strh * (shapes.ih - 1) + output_padding[1] + ((shapes.kh - 1) * attrs.dilh + 1) -
                               (attrs.padu + attrs.padd);
            // 2: w dim
            int64_t output_w = attrs.strw * (shapes.iw - 1) + output_padding[2] + ((shapes.kw - 1) * attrs.dilw + 1) -
                               (attrs.padl + attrs.padr);

            y_shape->SetDimNum(0);
            if (x_format == ge::FORMAT_NCDHW) {
                y_shape->AppendDim(shapes.in);
                y_shape->AppendDim(shapes.kc * attrs.groups);
                y_shape->AppendDim(output_d);
                y_shape->AppendDim(output_h);
                y_shape->AppendDim(output_w);
            } else if (x_format == ge::FORMAT_NDHWC) {
                y_shape->AppendDim(shapes.in);
                y_shape->AppendDim(output_d);
                y_shape->AppendDim(output_h);
                y_shape->AppendDim(output_w);
                y_shape->AppendDim(shapes.kc * attrs.groups);
            } else {
                OP_LOGE(
                    context->GetNodeName(), "The format of output y not support format %s.",
                    ge::TypeUtils::FormatToAscendString(x_format).GetString());
                return false;
            }

            OP_LOGD(context->GetNodeName(), "y_shape: %s", Ops::Base::ToString(*y_shape).c_str());
            return ge::GRAPH_SUCCESS;
        }

        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace Conv

IMPL_OP_INFERSHAPE(Conv3DTransposeV2)
    .InferShape(Ops::NN::Conv::InferShapeForConv3DTranspose)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvTransposeV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);

}  // namespace NN
}  // namespace Ops
