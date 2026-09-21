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
 * \file conv3d_transpose_infershape.cpp
 * \brief InferShape registration for legacy Conv3DTranspose, which is converted to
 *        Conv3DTransposeV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include "common/op_host/conv_backprop_infershape.h"
#include "common/op_host/conv_common_cube_util.h"

#include <algorithm>
#include <cstring>
#include <util/shape_util.h>

namespace Ops {
namespace NN {
namespace Conv {

using ge::Format;
using ge::FORMAT_DHWCN;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using gert::InferShapeContext;

static constexpr size_t kConv2dDimSizeLimit = 4;
static constexpr size_t kConv3dDimSizeLimit = 5;
static constexpr int32_t UNKNOWN_SHAPE_DIM = -1;

// Conv3DTranspose原型属性顺序(op_nn_proto_extend.h REG_OP)：strides(0) pads(1) dilations(2) groups(3)
// data_format(4) output_padding(5) offset_x(6)；
// 叠加IMPL_OP_INFERSHAPE注册的PrivateAttr：padding(7) _op_impl_mode_enum(8)
constexpr size_t kStridesIdx = 0;
constexpr size_t kPadsIdx = 1;
constexpr size_t kDilationsIdx = 2;
constexpr size_t kGroupsIdx = 3;
constexpr size_t kOutputPaddingIdx = 5;
constexpr size_t kPaddingIdx = 7;
constexpr size_t kInputSizeIndex = 0;
constexpr size_t kXIndex = 1;
constexpr size_t kFilterIndex = 2;

// onnx/torch适配层合成的input_size恒为全0占位符，全0时不采信const值，按反卷积公式重算输出shape：
// outD/H/W = stride*(x-1) + output_padding + (filter-1)*dilation + 1 - (pad_head+pad_tail等)；
// N取x的N维，C取filter的C维*groups
static bool GetConv3DXShapeForTranspose(const InferShapeContext* context, Format x_format, Conv3DInputShapes& shapes)
{
    const auto x_shape = context->GetInputShape(kXIndex);
    OP_CHECK_IF(x_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get x shape."),
                return false);
    OP_CHECK_IF(x_shape->GetDimNum() != kConv3dDimSizeLimit,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(x_shape->GetDimNum()).c_str(),
                                             std::to_string(kConv3dDimSizeLimit).c_str()),
                return false);

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
    } else {
        OP_LOGE(context->GetNodeName(), "The format of input x not support format %s.",
                ge::TypeUtils::FormatToAscendString(x_format).GetString());
        return false;
    }

    return true;
}

static bool GetConv3DFilterShapeForTranspose(const InferShapeContext* context, Conv3DInputShapes& shapes)
{
    const auto filter_desc = context->GetInputDesc(kFilterIndex);
    OP_CHECK_IF(filter_desc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter tensor desc."), return false);
    const auto filter_format = filter_desc->GetOriginFormat();
    const auto filter_shape = context->GetInputShape(kFilterIndex);
    OP_CHECK_IF(filter_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape."),
                return false);
    OP_CHECK_IF(filter_shape->GetDimNum() != kConv3dDimSizeLimit,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter",
                                             std::to_string(filter_shape->GetDimNum()).c_str(),
                                             std::to_string(kConv3dDimSizeLimit).c_str()),
                return false);

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
        OP_LOGE(context->GetNodeName(), "The format of input filter not support format %s.",
                ge::TypeUtils::FormatToAscendString(filter_format).GetString());
        return false;
    }

    return true;
}

static bool GetConv3DTransposeAttrs(const InferShapeContext* context, Format x_format, Conv3DInputShapes& shapes,
                                    Conv3DAttrs& attrs, int64_t output_padding[3])
{
    const auto runtime_attrs = context->GetAttrs();
    OP_CHECK_IF(runtime_attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs."),
                return false);
    const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kStridesIdx);
    const auto dilations_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kDilationsIdx);
    const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kPadsIdx);
    const auto output_padding_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kOutputPaddingIdx);
    const int64_t* groups = runtime_attrs->GetAttrPointer<int64_t>(kGroupsIdx);
    OP_CHECK_IF(strides_list == nullptr || dilations_list == nullptr || pads_list == nullptr ||
                    output_padding_list == nullptr || groups == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get strides/pads/dilations/groups attrs."),
                return false);
    OP_CHECK_IF(strides_list->GetSize() != kConv3dDimSizeLimit || dilations_list->GetSize() != kConv3dDimSizeLimit ||
                    output_padding_list->GetSize() != kConv3dDimSizeLimit || pads_list->GetSize() != 6,
                OP_LOGE(context->GetNodeName(),
                        "strides/dilations/output_padding size should be 5 and pads size should be 6 when input_size "
                        "is all-zero placeholder."),
                return false);

    const auto strides = static_cast<const int64_t*>(strides_list->GetData());
    const auto dilations = static_cast<const int64_t*>(dilations_list->GetData());
    const auto pads = static_cast<const int64_t*>(pads_list->GetData());
    const auto output_padding_data = static_cast<const int64_t*>(output_padding_list->GetData());
    size_t idx = 0;
    if (x_format == FORMAT_NCDHW) {
        attrs.strd = strides[kDDimNCDHWIdx];
        attrs.strh = strides[kHDimNCDHWIdx];
        attrs.strw = strides[kWDimNCDHWIdx];
        attrs.dild = dilations[kDDimNCDHWIdx];
        attrs.dilh = dilations[kHDimNCDHWIdx];
        attrs.dilw = dilations[kWDimNCDHWIdx];
        output_padding[idx++] = output_padding_data[kDDimNCDHWIdx];
        output_padding[idx++] = output_padding_data[kHDimNCDHWIdx];
        output_padding[idx++] = output_padding_data[kWDimNCDHWIdx];
    } else {
        // FORMAT_NDHWC, already checked in GetConv3DXShapeForTranspose, else is enough
        attrs.strd = strides[kDDimNDHWCIdx];
        attrs.strh = strides[kHDimNDHWCIdx];
        attrs.strw = strides[kWDimNDHWCIdx];
        attrs.dild = dilations[kDDimNDHWCIdx];
        attrs.dilh = dilations[kHDimNDHWCIdx];
        attrs.dilw = dilations[kWDimNDHWCIdx];
        output_padding[idx++] = output_padding_data[kDDimNDHWCIdx];
        output_padding[idx++] = output_padding_data[kHDimNDHWCIdx];
        output_padding[idx++] = output_padding_data[kWDimNDHWCIdx];
    }
    attrs.groups = *groups;
    OP_CHECK_IF(attrs.strd <= 0 || attrs.strh <= 0 || attrs.strw <= 0 || attrs.dild <= 0 || attrs.dilh <= 0 ||
                    attrs.dilw <= 0 || attrs.groups <= 0,
                OP_LOGE(context->GetNodeName(),
                        "strides, dilations and groups should be positive when input_size is all-zero placeholder."),
                return false);

    // pads布局为[head, tail, up, down, left, right]，与1.0 kConv3dPadHeadIdx等下标一致
    attrs.padf = pads[0];
    attrs.padb = pads[1];
    attrs.padu = pads[2];
    attrs.padd = pads[3];
    attrs.padl = pads[4];
    attrs.padr = pads[5];
    if (runtime_attrs->GetAttrNum() > kPaddingIdx) {
        const auto padding = runtime_attrs->GetAttrPointer<char>(kPaddingIdx);
        if (padding != nullptr && (strcmp(padding, "SAME") == 0)) {
            OP_LOGD(context->GetNodeName(), "get padding SAME.");
            constexpr int64_t paddingHalfDivisor = 2;
            int64_t tails_d = shapes.id % attrs.strd;
            int64_t tails_h = shapes.ih % attrs.strh;
            int64_t tails_w = shapes.iw % attrs.strw;
            int64_t dilate_kernel_d = attrs.dild * (shapes.kd - 1) + 1;
            int64_t dilate_kernel_h = attrs.dilh * (shapes.kh - 1) + 1;
            int64_t dilate_kernel_w = attrs.dilw * (shapes.kw - 1) + 1;
            int64_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - attrs.strd), 0L);
            int64_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.strh), 0L);
            int64_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.strw), 0L);
            attrs.padf = pad_d / paddingHalfDivisor;
            attrs.padb = pad_d - attrs.padf;
            attrs.padu = pad_h / paddingHalfDivisor;
            attrs.padd = pad_h - attrs.padu;
            attrs.padl = pad_w / paddingHalfDivisor;
            attrs.padr = pad_w - attrs.padl;
        }
    }
    return true;
}

static ge::graphStatus FixZeroPlaceholderShape(gert::InferShapeContext* context, bool from_2d, int32_t d_index)
{
    const auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get y shape."),
                return ge::GRAPH_FAILED);
    if (!CheckOutputAllZero(y_shape) && !(from_2d && CheckOutputAllZeroFrom2D(context, y_shape, d_index))) {
        return ge::GRAPH_SUCCESS;
    }

    const auto x_desc = context->GetInputDesc(kXIndex);
    OP_CHECK_IF(x_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x desc is null"),
                return ge::GRAPH_FAILED);
    const auto x_format = x_desc->GetOriginFormat();

    Conv3DInputShapes shapes;
    Conv3DAttrs attrs;
    int64_t output_padding[3]; // 3: DHW
    OP_CHECK_IF(!GetConv3DXShapeForTranspose(context, x_format, shapes) ||
                    !GetConv3DFilterShapeForTranspose(context, shapes) ||
                    !GetConv3DTransposeAttrs(context, x_format, shapes, attrs, output_padding),
                OP_LOGE(context->GetNodeName(), "failed to get x/filter/attrs for all-zero placeholder."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shapes.in <= 0 || shapes.id <= 0 || shapes.ih <= 0 || shapes.iw <= 0 || shapes.kc <= 0 ||
                    shapes.kd <= 0 || shapes.kh <= 0 || shapes.kw <= 0,
                OP_LOGE(context->GetNodeName(),
                        "x shape and filter shape should be positive when input_size is all-zero placeholder."),
                return ge::GRAPH_FAILED);

    int64_t output_d = attrs.strd * (shapes.id - 1) + output_padding[0] + ((shapes.kd - 1) * attrs.dild + 1) -
                       (attrs.padf + attrs.padb);
    int64_t output_h = attrs.strh * (shapes.ih - 1) + output_padding[1] + ((shapes.kh - 1) * attrs.dilh + 1) -
                       (attrs.padu + attrs.padd);
    int64_t output_w = attrs.strw * (shapes.iw - 1) + output_padding[2] + ((shapes.kw - 1) * attrs.dilw + 1) -
                       (attrs.padl + attrs.padr);
    OP_CHECK_IF(output_d <= 0 || output_h <= 0 || output_w <= 0,
                OP_LOGE(context->GetNodeName(),
                        "inferred output shape d = %ld, h = %ld, w = %ld should be positive when input_size is "
                        "all-zero placeholder.",
                        output_d, output_h, output_w),
                return ge::GRAPH_FAILED);

    y_shape->SetDimNum(0);
    if (x_format == FORMAT_NCDHW) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(shapes.kc * attrs.groups);
        y_shape->AppendDim(output_d);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
    } else if (x_format == FORMAT_NDHWC) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(output_d);
        y_shape->AppendDim(output_h);
        y_shape->AppendDim(output_w);
        y_shape->AppendDim(shapes.kc * attrs.groups);
    } else {
        OP_LOGE(context->GetNodeName(), "The format of output y not support format %s.",
                ge::TypeUtils::FormatToAscendString(x_format).GetString());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Conv3DTranspose(gert::InferShapeContext* context)
{
    auto const_tensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get null tensor"),
                return ge::GRAPH_FAILED);
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());

    auto ret = ge::GRAPH_SUCCESS;
    if (const_tensor_dim_num == kConv2dDimSizeLimit) {
        ret = InferShapeForConvBackpropExtend3D(context, 0, "input_size");
    } else {
        ret = InferShapeForConvBackprop(context, 0, "input_size", kConv3dDimSizeLimit);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        const auto y_desc = context->GetOutputDesc(0);
        int32_t d_index = UNKNOWN_SHAPE_DIM;
        if (y_desc != nullptr) {
            d_index = GetConvBackpropIndex(y_desc->GetOriginFormat());
        }
        ret = FixZeroPlaceholderShape(context, const_tensor_dim_num == kConv2dDimSizeLimit, d_index);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        auto yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "[InferShape] Conv3DTranspose y_shape: %s",
                    Ops::Base::ToString(*yShape).c_str());
        }
    }
    return ret;
}

} // namespace Conv

IMPL_OP_INFERSHAPE(Conv3DTranspose)
    .InferShape(Ops::NN::Conv::InferShape4Conv3DTranspose)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvTransposeV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);

} // namespace NN
} // namespace Ops
