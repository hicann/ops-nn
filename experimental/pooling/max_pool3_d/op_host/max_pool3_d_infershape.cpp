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
 * \file max_pool_3d_infershape.cpp
 * \brief
 */
#include <algorithm>
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
constexpr size_t SHAPE_SIZE = 5;
constexpr size_t PAD_SIZE = 6;
constexpr size_t PAD_FRONT = 0;
constexpr size_t PAD_BACK = 1;
constexpr size_t PAD_TOP = 2;
constexpr size_t PAD_BOTTOM = 3;
constexpr size_t PAD_LEFT = 4;
constexpr size_t PAD_RIGHT = 5;

using InferShapePaddingFunc = ge::graphStatus (*)(gert::InferShapeContext*, size_t, size_t, size_t,
                                                  const gert::RuntimeAttrs*);

struct SpatialValues {
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
};

static ge::graphStatus GetSpatialValues(gert::InferShapeContext* context, const gert::RuntimeAttrs* attrs,
                                        size_t attrIndex, size_t dDim, size_t hDim, size_t wDim, const char* attrName,
                                        bool requireOuterDimsOne, SpatialValues& values)
{
    const auto* attr = attrs->GetAttrPointer<gert::ContinuousVector>(attrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const size_t size = attr->GetSize();
    OP_CHECK_IF(size != 1U && size != 3U && size != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "Length of %s %zu must be 1, 3, or 5.", attrName, size),
                return ge::GRAPH_FAILED);
    const auto* data = static_cast<const int64_t*>(attr->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, data);

    if (size == 1U) {
        values = {data[0], data[0], data[0]};
    } else if (size == 3U) {
        values = {data[0], data[1], data[2]};
    } else {
        values = {data[dDim], data[hDim], data[wDim]};
        const size_t cDim = dDim == 1U ? 4U : 1U;
        OP_CHECK_IF(requireOuterDimsOne && (data[0] != 1 || data[cDim] != 1),
                    OP_LOGE(context->GetNodeName(), "%s values on N and C dimensions must be 1.", attrName),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(values.d <= 0 || values.h <= 0 || values.w <= 0,
                OP_LOGE(context->GetNodeName(), "%s spatial values [%ld, %ld, %ld] must be greater than 0.", attrName,
                        values.d, values.h, values.w),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t InferSameOutputDim(int64_t dimSize, int64_t stride)
{
    if (stride == 0) {
        return 0;
    }
    return (dimSize + stride - 1) / stride;
}

static int64_t FloorDiv(int64_t dividend, int64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    int64_t quotient = dividend / divisor;
    if (dividend % divisor < 0) {
        --quotient;
    }
    return quotient;
}

static int64_t InferCalculatedOutputDim(int64_t dimSize, int64_t ksize, int64_t stride, int64_t dilation, bool ceilMode)
{
    int64_t numerator = dimSize - dilation * (ksize - 1) - 1;
    if (ceilMode) {
        numerator += stride - 1;
    }
    return FloorDiv(numerator, stride) + 1;
}

static ge::graphStatus InferShapePaddingCalculated(gert::InferShapeContext* context, size_t d_dim, size_t h_dim,
                                                   size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    SpatialValues ksize;
    SpatialValues strides;
    SpatialValues dilations;
    OP_CHECK_IF(
        GetSpatialValues(context, attrs, INDEX_KSIZE, d_dim, h_dim, w_dim, "ksize", true, ksize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to parse ksize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_STRIDES, d_dim, h_dim, w_dim, "strides", true, strides) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse strides."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_DILATION, d_dim, h_dim, w_dim, "dilation", false, dilations) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse dilation."), return ge::GRAPH_FAILED);

    const auto* pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(pads->GetSize() != PAD_SIZE,
                OP_LOGE(context->GetNodeName(), "Length of pads %zu must be 6!", pads->GetSize()), return GRAPH_FAILED);
    const auto* padsData = static_cast<const int64_t*>(pads->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, padsData);
    for (size_t index = 0U; index < PAD_SIZE; ++index) {
        OP_CHECK_IF(
            padsData[index] < 0,
            OP_LOGE(context->GetNodeName(), "pads[%zu] must be nonnegative, but got %ld.", index, padsData[index]),
            return ge::GRAPH_FAILED);
    }
    const auto* ceil_mode = attrs->GetAttrPointer<int32_t>(INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;
    int64_t dim_size = in_shape->GetDim(d_dim);
    int64_t out_dim_size = InferCalculatedOutputDim(dim_size + padsData[PAD_FRONT] + padsData[PAD_BACK], ksize.d,
                                                    strides.d, dilations.d, *ceil_mode != 0);
    if (*ceil_mode != 0 && (out_dim_size - 1) * strides.d >= dim_size + padsData[PAD_FRONT]) {
        --out_dim_size;
    }
    out_shape->SetDim(d_dim, out_dim_size);

    dim_size = in_shape->GetDim(h_dim);
    out_dim_size = InferCalculatedOutputDim(dim_size + padsData[PAD_TOP] + padsData[PAD_BOTTOM], ksize.h, strides.h,
                                            dilations.h, *ceil_mode != 0);
    if (*ceil_mode != 0 && (out_dim_size - 1) * strides.h >= dim_size + padsData[PAD_TOP]) {
        --out_dim_size;
    }
    out_shape->SetDim(h_dim, out_dim_size);

    dim_size = in_shape->GetDim(w_dim);
    out_dim_size = InferCalculatedOutputDim(dim_size + padsData[PAD_LEFT] + padsData[PAD_RIGHT], ksize.w, strides.w,
                                            dilations.w, *ceil_mode != 0);
    if (*ceil_mode != 0 && (out_dim_size - 1) * strides.w >= dim_size + padsData[PAD_LEFT]) {
        --out_dim_size;
    }
    out_shape->SetDim(w_dim, out_dim_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingValid(gert::InferShapeContext* context, size_t d_dim, size_t h_dim,
                                              size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    SpatialValues ksize;
    SpatialValues strides;
    SpatialValues dilations;
    OP_CHECK_IF(
        GetSpatialValues(context, attrs, INDEX_KSIZE, d_dim, h_dim, w_dim, "ksize", true, ksize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to parse ksize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_STRIDES, d_dim, h_dim, w_dim, "strides", true, strides) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse strides."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_DILATION, d_dim, h_dim, w_dim, "dilation", false, dilations) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse dilation."), return ge::GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    out_shape->SetDim(d_dim, InferCalculatedOutputDim(in_shape->GetDim(d_dim), ksize.d, strides.d, dilations.d, false));
    out_shape->SetDim(h_dim, InferCalculatedOutputDim(in_shape->GetDim(h_dim), ksize.h, strides.h, dilations.h, false));
    out_shape->SetDim(w_dim, InferCalculatedOutputDim(in_shape->GetDim(w_dim), ksize.w, strides.w, dilations.w, false));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(gert::InferShapeContext* context, size_t d_dim, size_t h_dim, size_t w_dim,
                                             const gert::RuntimeAttrs* attrs)
{
    SpatialValues ksize;
    SpatialValues strides;
    SpatialValues dilations;
    OP_CHECK_IF(
        GetSpatialValues(context, attrs, INDEX_KSIZE, d_dim, h_dim, w_dim, "ksize", true, ksize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to parse ksize."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_STRIDES, d_dim, h_dim, w_dim, "strides", true, strides) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse strides."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSpatialValues(context, attrs, INDEX_DILATION, d_dim, h_dim, w_dim, "dilation", false, dilations) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse dilation."), return ge::GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    out_shape->SetDim(d_dim, InferSameOutputDim(in_shape->GetDim(d_dim), strides.d));
    out_shape->SetDim(h_dim, InferSameOutputDim(in_shape->GetDim(h_dim), strides.h));
    out_shape->SetDim(w_dim, InferSameOutputDim(in_shape->GetDim(w_dim), strides.w));

    return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"CALCULATED", InferShapePaddingCalculated},
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

static ge::graphStatus InferShape4MaxPool3D(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MaxPool3D", "Infer-shape context is null.");
        return GRAPH_FAILED;
    }
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

    OP_CHECK_IF(xShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "MaxPool3D input rank must be 5, but got %zu.", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const auto* src_td = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, src_td);
    const ge::Format input_format = src_td->GetOriginFormat();
    OP_CHECK_IF(input_format != FORMAT_NDHWC && input_format != FORMAT_NCDHW,
                OP_LOGE(context->GetNodeName(), "MaxPool3D origin format must be NCDHW or NDHWC, but got %d.",
                        static_cast<int32_t>(input_format)),
                return ge::GRAPH_FAILED);
    const size_t d_dim = input_format == FORMAT_NDHWC ? 1U : 2U;
    const size_t h_dim = input_format == FORMAT_NDHWC ? 2U : 3U;
    const size_t w_dim = input_format == FORMAT_NDHWC ? 3U : 4U;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding_mode);
    auto it = std::find_if(kFuncMap.begin(), kFuncMap.end(),
                           [&padding_mode](const std::pair<std::string, InferShapePaddingFunc>& item) -> bool {
                               return item.first == padding_mode;
                           });
    OP_CHECK_IF(it == kFuncMap.end(),
                OP_LOGE(context->GetNodeName(), "padding_mode %s must in (CALCULATED, VALID, SAME).", padding_mode),
                return GRAPH_FAILED);

    return it->second(context, d_dim, h_dim, w_dim, attrs);
}

static ge::graphStatus InferDataTypeForMaxPool3D(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MaxPool3D", "Infer-dtype context is null.");
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3D).InferShape(InferShape4MaxPool3D).InferDataType(InferDataTypeForMaxPool3D);
} // namespace ops
