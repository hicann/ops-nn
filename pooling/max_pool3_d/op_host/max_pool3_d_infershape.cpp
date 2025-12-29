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
 * \file max_pool_3d_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_KSIZE = 0;
constexpr size_t INDEX_STRIDES = 1;
constexpr size_t INDEX_PADDING = 2;
constexpr size_t INDEX_PADS = 3;
constexpr size_t INDEX_DILATION = 4;
constexpr size_t INDEX_CEIL_MODE = 5;
constexpr size_t INDEX_DATA_FORMAT = 6;
constexpr size_t SHAPE_SIZE = 5;
constexpr size_t DIM_SIZE6 = 6;
constexpr size_t PAD_SIZE = 6;
constexpr size_t PAD_FRONT = 0;
constexpr size_t PAD_BACK = 1;
constexpr size_t PAD_TOP = 2;
constexpr size_t PAD_BOTTOM = 3;
constexpr size_t PAD_LEFT = 4;
constexpr size_t PAD_RIGHT = 5;

typedef ge::graphStatus (*InferShapePaddingFunc)(
    gert::InferShapeContext*, size_t, size_t, size_t, const gert::RuntimeAttrs*);

static std::string Int64ToString(const int64_t* data, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(data[i]) + " ";
    }
    r = r + "]";
    return r;
}

static int64_t SameUpdateDim(const int64_t ksize, const int64_t strides, int64_t dim_size)
{
    return (strides == 0) ? (dim_size - ksize + 1) : ((dim_size - ksize + strides) / strides);
}

static void CalculateUpdateDim(
    const int64_t ksize, const int64_t stride, const int64_t dilation, const int32_t ceil_mode, int64_t& dim_size)
{
    if (ceil_mode) {
        dim_size = (stride == 0) ? (dim_size - ksize + 1) :
                                   (dim_size - dilation * (ksize - 1) - 1 + stride + stride - 1) / stride;
    } else {
        dim_size = (stride == 0) ? (dim_size - ksize + 1) : (dim_size - dilation * (ksize - 1) - 1 + stride) / stride;
    }
}

static ge::graphStatus InferShapePaddingCalculated(
    gert::InferShapeContext* context, size_t d_dim, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %zu must be 5!", ksize->GetSize()), return GRAPH_FAILED);
    auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        pads->GetSize() != PAD_SIZE, OP_LOGE(context->GetNodeName(), "Length of pads %zu must be 6!", pads->GetSize()),
        return GRAPH_FAILED);
    auto ceil_mode = attrs->GetAttrPointer<int32_t>(INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 5!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[d_dim] <= 0 || strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %lld and w %lld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[d_dim], strides_data[h_dim],
            strides_data[w_dim]),
        return GRAPH_FAILED);
    auto pads_data = reinterpret_cast<const int64_t*>(pads->GetData());

    auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_DILATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilations);
    OP_CHECK_IF(
        ((dilations->GetSize() != SHAPE_SIZE) && (dilations->GetSize() != DIM_SIZE6)),
        OP_LOGE(context->GetNodeName(), "Length of dilation %zu must be 5!", dilations->GetSize()),
        return GRAPH_FAILED);
    auto dilations_data = reinterpret_cast<const int64_t*>(dilations->GetData());
    if (dilations->GetSize() == DIM_SIZE6) {
        OP_LOGW("InferShapePaddingCalculated", "The size of dilationList: 6 is deprecated, should be 5");
    }

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;
    int64_t dim_size = in_shape->GetDim(d_dim);
    int64_t out_dim_size = dim_size + pads_data[PAD_FRONT] + pads_data[PAD_BACK];
    CalculateUpdateDim(ksize_data[d_dim], strides_data[d_dim], dilations_data[d_dim], *ceil_mode, out_dim_size);
    if ((out_dim_size - 1) * strides_data[d_dim] >= dim_size + pads_data[PAD_FRONT]) {
        out_dim_size = out_dim_size - 1;
    }
    out_shape->SetDim(d_dim, out_dim_size);

    dim_size = in_shape->GetDim(h_dim);
    out_dim_size = dim_size + pads_data[PAD_TOP] + pads_data[PAD_BOTTOM];
    CalculateUpdateDim(ksize_data[h_dim], strides_data[h_dim], dilations_data[h_dim], *ceil_mode, out_dim_size);
    if ((out_dim_size - 1) * strides_data[h_dim] >= dim_size + pads_data[PAD_TOP]) {
        out_dim_size = out_dim_size - 1;
    }
    out_shape->SetDim(h_dim, out_dim_size);

    dim_size = in_shape->GetDim(w_dim);
    out_dim_size = dim_size + pads_data[PAD_LEFT] + pads_data[PAD_RIGHT];
    CalculateUpdateDim(ksize_data[w_dim], strides_data[w_dim], dilations_data[w_dim], *ceil_mode, out_dim_size);
    if ((out_dim_size - 1) * strides_data[w_dim] >= dim_size + pads_data[PAD_LEFT]) {
        out_dim_size = out_dim_size - 1;
    }
    out_shape->SetDim(w_dim, out_dim_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingValid(
    gert::InferShapeContext* context, size_t d_dim, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %zu must be 5!", ksize->GetSize()), return GRAPH_FAILED);
    auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 5!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[d_dim] <= 0 || strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %lld and w %lld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[d_dim], strides_data[h_dim],
            strides_data[w_dim]),
        return GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(d_dim);
    out_shape->SetDim(d_dim, SameUpdateDim(ksize_data[d_dim], strides_data[d_dim], dim_size));
    dim_size = in_shape->GetDim(h_dim);
    out_shape->SetDim(h_dim, SameUpdateDim(ksize_data[h_dim], strides_data[h_dim], dim_size));
    dim_size = in_shape->GetDim(w_dim);
    out_shape->SetDim(w_dim, SameUpdateDim(ksize_data[w_dim], strides_data[w_dim], dim_size));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(
    gert::InferShapeContext* context, size_t d_dim, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 5!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[d_dim] <= 0 || strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %lld and w %lld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[d_dim], strides_data[h_dim],
            strides_data[w_dim]),
        return GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(d_dim);
    out_shape->SetDim(d_dim, SameUpdateDim(1, strides_data[d_dim], dim_size));
    dim_size = in_shape->GetDim(h_dim);
    out_shape->SetDim(h_dim, SameUpdateDim(1, strides_data[h_dim], dim_size));
    dim_size = in_shape->GetDim(w_dim);
    out_shape->SetDim(w_dim, SameUpdateDim(1, strides_data[w_dim], dim_size));

    return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"CALCULATED", InferShapePaddingCalculated},
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

static ge::graphStatus InferShape4MaxPool3D(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownShape(*xShape)) {
        Ops::Base::SetUnknownShape(xShape->GetDimNum(), *yShape);
        return ge::GRAPH_SUCCESS;
    }

    auto src_td = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, src_td);
    auto input_format = src_td->GetOriginFormat();
    size_t d_dim = input_format == FORMAT_NDHWC ? 1 : 2;
    size_t h_dim = input_format == FORMAT_NDHWC ? 2 : 3;
    size_t w_dim = input_format == FORMAT_NDHWC ? 3 : 4;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding_mode);
    auto it = std::find_if(
        kFuncMap.begin(), kFuncMap.end(),
        [&padding_mode](const std::pair<std::string, InferShapePaddingFunc>& item) -> bool {
            return item.first == padding_mode;
        });
    OP_CHECK_IF(
        it == kFuncMap.end(),
        OP_LOGE(context->GetNodeName(), "padding_mode %s must in (CALCULATED, VALID, SAME).", padding_mode),
        return GRAPH_FAILED);

    // when padding_mode in (CALCULATED, VALID, SAME)
    return it->second(context, d_dim, h_dim, w_dim, attrs);
}

static ge::graphStatus InferDataTypeForMaxPool3D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3D).InferShape(InferShape4MaxPool3D).InferDataType(InferDataTypeForMaxPool3D);
} // namespace ops
