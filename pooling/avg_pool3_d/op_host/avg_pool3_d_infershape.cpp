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
 * \file avg_pool3_d_infershape.cpp
 * \brief
 */

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace {
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t DIM_5 = 5;
constexpr size_t LEN_1 = 1;
constexpr size_t LEN_3 = 3;
constexpr size_t LEN_6 = 6;
constexpr size_t kConv3dInputSizeLimit = 5;

// NDHWC
constexpr size_t kDDimNDHWCIdx = 1;
constexpr size_t kHDimNDHWCIdx = 2;
constexpr size_t kWDimNDHWCIdx = 3;

// AvgPool3D
constexpr size_t kDHWSizeLimit = 3;
constexpr size_t kAvgPool3DKSizeIdx = 0;
constexpr size_t kAvgPool3DStridesIdx = 1;
constexpr size_t kAvgPool3DPadsIdx = 2;
constexpr size_t kAvgPool3DCeilModeIdx = 3;
constexpr size_t kAvgPool3DPaddingIdx = 7;

// DHWCN
constexpr size_t kDDimDHWCNIdx = 0;
constexpr size_t kHDimDHWCNIdx = 1;
constexpr size_t kWDimDHWCNIdx = 2;

// NCDHW
constexpr size_t kDDimNCDHWIdx = 2;
constexpr size_t kHDimNCDHWIdx = 3;
constexpr size_t kWDimNCDHWIdx = 4;

using gert::InferShapeContext;
}  // namespace

using namespace ge;

namespace ops {
using ge::Format;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_DHWCN;
struct Conv3DInputShapesForAvgPool3d {
    int64_t in = 0;
    int64_t id = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t ic = 0;

    int64_t kn = 0;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t kc = 0;
};

struct Conv3DAttrsForAvgPool3d {
    bool ceil_mode = false; // only for AvgPool3D

    int64_t strd = 0;
    int64_t strh = 0;
    int64_t strw = 0;

    int64_t dild = 1;
    int64_t dilh = 1;
    int64_t dilw = 1;

    int64_t groups = 1;

    int64_t padf = 0;
    int64_t padb = 0;
    int64_t padu = 0;
    int64_t padd = 0;
    int64_t padl = 0;
    int64_t padr = 0;
};

static bool GetConv3DXShape(
    const InferShapeContext *context, size_t x_idx, Format x_format, bool avg_pool3d, Conv3DInputShapesForAvgPool3d &shapes) {
    const auto x_shape = context->GetInputShape(x_idx);
    OP_LOGE_IF(x_shape == nullptr, false, context->GetNodeName(), "failed to get x shape.");
    OP_LOGE_IF(x_shape->GetDimNum() != kConv3dInputSizeLimit, false, context->GetNodeName(),
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
        OP_LOGE(context->GetNodeName(), "The format of input x not support format %s.",
                ge::TypeUtils::FormatToAscendString(x_format).GetString());
        return false;
    }

    return true;
}

static bool GetConv3DPads(
    const InferShapeContext *context, const Conv3DInputShapesForAvgPool3d &shapes, size_t pads_idx, size_t padding_idx,
    Conv3DAttrsForAvgPool3d &attrs) {
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");
    const auto pads_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(pads_idx);
    OP_LOGE_IF(pads_list == nullptr, false, context->GetNodeName(), "failed to get pads attrs");
    OP_LOGE_IF(pads_list->GetSize() != LEN_6 and pads_list->GetSize() != LEN_3 and pads_list->GetSize() != LEN_1,
               false, context->GetNodeName(), "invalid pads");
    const auto pads_list_data = static_cast<const int64_t *>(pads_list->GetData());

    if (pads_list->GetSize() == LEN_1 || pads_list->GetSize() == LEN_3){
        attrs.padf = pads_list_data[DIM_0];
        attrs.padb = pads_list_data[DIM_0];
        attrs.padu = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_1];
        attrs.padd = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_1];
        attrs.padl = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_2];
        attrs.padr = pads_list->GetSize() == LEN_1 ? pads_list_data[DIM_0] : pads_list_data[DIM_2];
    }else{
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
            int64_t tails_d = shapes.id % attrs.strd;  // non zero, checked in shape range infer logic
            int64_t tails_h = shapes.ih % attrs.strh;  // non zero, checked in shape range infer logic
            int64_t tails_w = shapes.iw % attrs.strw;  // non zero, checked in shape range infer logic
            int64_t dilate_kernel_d = attrs.dild * (shapes.kd - 1) + 1;
            int64_t dilate_kernel_h = attrs.dilh * (shapes.kh - 1) + 1;
            int64_t dilate_kernel_w = attrs.dilw * (shapes.kw - 1) + 1;
            int64_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - attrs.strd), 0L);
            int64_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - attrs.strh), 0L);
            int64_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - attrs.strw), 0L);
            attrs.padf = pad_d / 2;  // 2 means pad_up is half size of pad_d
            attrs.padb = pad_d - attrs.padf;
            attrs.padu = pad_h / 2;  // 2 means pad_up is half size of pad_h
            attrs.padd = pad_h - attrs.padu;
            attrs.padl = pad_w / 2;  // 2 means pad_up is half size of pad_w
            attrs.padr = pad_w - attrs.padl;
            return true;
        }
    }

    bool negative_pad =
        std::any_of(pads_list_data, pads_list_data + pads_list->GetSize(), [](int64_t val) -> bool { return val < 0; });
    OP_LOGE_IF(negative_pad, false, context->GetNodeName(), "The value of the pads attribute should >= 0");

    return true;
}

static bool GetStridesForAvgPool3D(
    const InferShapeContext *context, ge::Format x_format, Conv3DAttrsForAvgPool3d &attrs) {
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");

    const auto strides_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kAvgPool3DStridesIdx);
    OP_LOGE_IF(strides_list == nullptr, false, context->GetNodeName(), "failed to get strides attrs");
    const auto strides_list_data = static_cast<const int64_t *>(strides_list->GetData());

    if (strides_list->GetSize() == 1) {
        attrs.strd = strides_list_data[0];
        attrs.strh = strides_list_data[0];
        attrs.strw = strides_list_data[0];
    } else if (strides_list->GetSize() == kDHWSizeLimit) {
        size_t idx = 0;
        attrs.strd = strides_list_data[idx++];
        attrs.strh = strides_list_data[idx++];
        attrs.strw = strides_list_data[idx++];
    } else if (strides_list->GetSize() == kConv3dInputSizeLimit) {
        if (x_format == ge::FORMAT_NCDHW) {
            attrs.strd = strides_list_data[kDDimNCDHWIdx];
            attrs.strh = strides_list_data[kHDimNCDHWIdx];
            attrs.strw = strides_list_data[kWDimNCDHWIdx];
        } else if (x_format == ge::FORMAT_NDHWC) {
            attrs.strd = strides_list_data[kDDimNDHWCIdx];
            attrs.strh = strides_list_data[kHDimNDHWCIdx];
            attrs.strw = strides_list_data[kWDimNDHWCIdx];
        } else {
            // DHWCN
            attrs.strd = strides_list_data[kDDimDHWCNIdx];
            attrs.strh = strides_list_data[kHDimDHWCNIdx];
            attrs.strw = strides_list_data[kWDimDHWCNIdx];
        }
    }

    OP_LOGE_IF(attrs.strd == 0 || attrs.strh == 0 || attrs.strw == 0, false,
               context->GetNodeName(), "get zero strides");
    return true;
}

static bool GetKSize(const InferShapeContext *context, ge::Format x_format, Conv3DInputShapesForAvgPool3d &shapes) {
    const auto runtime_attrs = context->GetAttrs();
    OP_LOGE_IF(runtime_attrs == nullptr, false, context->GetNodeName(), "failed to get runtime attrs");

    const auto ksize_list = runtime_attrs->GetAttrPointer<gert::ContinuousVector>(kAvgPool3DKSizeIdx);
    OP_LOGE_IF(ksize_list == nullptr, false, context->GetNodeName(), "failed to get ksize attrs");
    const auto ksize_list_data = static_cast<const int64_t *>(ksize_list->GetData());

    if (ksize_list->GetSize() == 1) {
        shapes.kd = ksize_list_data[0];
        shapes.kh = ksize_list_data[0];
        shapes.kw = ksize_list_data[0];
    } else if (ksize_list->GetSize() == kDHWSizeLimit) {
        size_t idx = 0;
        shapes.kd = ksize_list_data[idx++];
        shapes.kh = ksize_list_data[idx++];
        shapes.kw = ksize_list_data[idx++];
    } else if (ksize_list->GetSize() == kConv3dInputSizeLimit) {
        if (x_format == ge::FORMAT_NCDHW) {
            shapes.kd = ksize_list_data[kDDimNCDHWIdx];
            shapes.kh = ksize_list_data[kHDimNCDHWIdx];
            shapes.kw = ksize_list_data[kWDimNCDHWIdx];
        } else if (x_format == ge::FORMAT_NDHWC) {
            shapes.kd = ksize_list_data[kDDimNDHWCIdx];
            shapes.kh = ksize_list_data[kHDimNDHWCIdx];
            shapes.kw = ksize_list_data[kWDimNDHWCIdx];
        } else {
            // DHWCN
            shapes.kd = ksize_list_data[kDDimDHWCNIdx];
            shapes.kh = ksize_list_data[kHDimDHWCNIdx];
            shapes.kw = ksize_list_data[kWDimDHWCNIdx];
        }
    }

    return true;
}


static bool CalcAvgPool3DOutputShape(
    const char *op_name, const Conv3DInputShapesForAvgPool3d &shapes, const Conv3DAttrsForAvgPool3d &attrs,
    ge::Format y_format, gert::Shape *y_shape) {
    int64_t outd = 0;
    int64_t outh = 0;
    int64_t outw = 0;
    if (attrs.ceil_mode) {
        outd = (shapes.id + attrs.padf + attrs.padb - shapes.kd + attrs.strd - 1) / attrs.strd + 1;
        outh = (shapes.ih + attrs.padu + attrs.padd - shapes.kh + attrs.strh - 1) / attrs.strh + 1;
        outw = (shapes.iw + attrs.padl + attrs.padr - shapes.kw + attrs.strw - 1) / attrs.strw + 1;
        if ((outd - 1) * attrs.strd >= shapes.id + attrs.padf) {
            outd--;
        }
        if ((outh - 1) * attrs.strh >= shapes.ih + attrs.padu) {
            outh--;
        }
        if ((outw - 1) * attrs.strw >= shapes.iw + attrs.padl) {
            outw--;
        }
    } else {
        outd = (shapes.id + attrs.padf + attrs.padb - shapes.kd) / attrs.strd + 1;
        outh = (shapes.ih + attrs.padu + attrs.padd - shapes.kh) / attrs.strh + 1;
        outw = (shapes.iw + attrs.padl + attrs.padr - shapes.kw) / attrs.strw + 1;
    }

    y_shape->SetDimNum(0);
    if (y_format == ge::FORMAT_NCDHW) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(shapes.ic);
        y_shape->AppendDim(outd);
        y_shape->AppendDim(outh);
        y_shape->AppendDim(outw);
    } else if (y_format == ge::FORMAT_NDHWC) {
        y_shape->AppendDim(shapes.in);
        y_shape->AppendDim(outd);
        y_shape->AppendDim(outh);
        y_shape->AppendDim(outw);
        y_shape->AppendDim(shapes.ic);
    } else if (y_format == ge::FORMAT_DHWCN) {
        y_shape->AppendDim(outd);
        y_shape->AppendDim(outh);
        y_shape->AppendDim(outw);
        y_shape->AppendDim(shapes.ic);
        y_shape->AppendDim(shapes.in);
    } else {
        OP_LOGE(op_name, "The format of output y not support format %s.",
                ge::TypeUtils::FormatToAscendString(y_format).GetString());
        return false;
    }

    return true;
}

static ge::graphStatus InferShapeForAvgPool3D(InferShapeContext *context) {
    if (context == nullptr) {
        OP_LOGE("AvgPool3D", "context is null");
        return ge::GRAPH_FAILED;
    }
    const auto op_name = context->GetNodeName() == nullptr ? "AvgPool3D" : context->GetNodeName();

    const auto runtime_attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, runtime_attrs);
    const auto ceil_mode = runtime_attrs->GetAttrPointer<bool>(kAvgPool3DCeilModeIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);

    const auto x_desc = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    const auto x_format = x_desc->GetOriginFormat();

    const auto y_shape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    const auto y_desc = context->GetOutputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_desc);
    const auto y_format = y_desc->GetOriginFormat();

    Conv3DInputShapesForAvgPool3d shapes;
    Conv3DAttrsForAvgPool3d attrs;
    attrs.ceil_mode = *ceil_mode;
    if (GetConv3DXShape(context, 0UL, x_format, true, shapes) && GetKSize(context, x_format, shapes) &&
        GetStridesForAvgPool3D(context, x_format, attrs) &&
        GetConv3DPads(context, shapes, kAvgPool3DPadsIdx, kAvgPool3DPaddingIdx, attrs) &&
        CalcAvgPool3DOutputShape(op_name, shapes, attrs, y_format, y_shape)) {
        OP_LOGD(context->GetNodeName(), "y_shape: %s", Shape2String(*y_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(op_name, "failed to infer shape for AvgPool3D");
    return ge::GRAPH_FAILED;
}

IMPL_OP_INFERSHAPE(AvgPool3D)
    .InferShape(InferShapeForAvgPool3D)
    .PrivateAttr("padding", "");
}  // namespace ops