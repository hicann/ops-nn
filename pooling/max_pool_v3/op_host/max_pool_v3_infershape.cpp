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
 * \file max_pool_v3.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
constexpr size_t INDEX_KSIZE = 0;
constexpr size_t INDEX_STRIDES = 1;
constexpr size_t INDEX_PADDING_MODE = 2;
constexpr size_t INDEX_PADS = 3;
constexpr size_t INDEX_DATA_FORMAT = 4;
constexpr size_t INDEX_GLOBAL_POOLING = 5;
constexpr size_t INDEX_CEIL_MODE = 6;
constexpr size_t SHAPE_NHWC_SIZE = 4;
constexpr size_t INSHAPE_DIM_NUM = 4;
constexpr size_t PAD_SIZE = 4;
constexpr size_t PAD_TOP = 0;
constexpr size_t PAD_BOTTOM = 1;
constexpr size_t PAD_LEFT = 2;
constexpr size_t PAD_RIGHT = 3;

using InferShapePaddingFunc = ge::graphStatus (*)(gert::InferShapeContext*, size_t, size_t, const gert::RuntimeAttrs*);

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
    if (dim_size == UNKNOWN_DIM_VALUE_) {
        return UNKNOWN_DIM_VALUE_;
    }
    return (strides == 0) ? (dim_size - ksize + 1) : ((dim_size - ksize + strides) / strides);
}

// Integer division rounding to -Infinity
static inline int64_t divRtn(const int64_t x, const int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    };
    return q;
}

// 计算经过MaxPool后的shape的h和w（n,c与input一致，不用计算）
static inline int64_t CalculateUpdateDim(
    const int64_t ksize, const int64_t padL, const int64_t padR, const int64_t stride, const bool ceil_mode,
    int64_t& dim_size)
{
    if (dim_size == UNKNOWN_DIM_VALUE_) {
        return UNKNOWN_DIM_VALUE_;
    }

    int64_t outputSize = divRtn(dim_size + padL + padR - ksize + (ceil_mode ? stride - 1 : 0), stride) + 1;

    if (ceil_mode) {
        if ((outputSize - 1) * stride >= dim_size + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

static ge::graphStatus InferShapeGlobalPooling(gert::InferShapeContext* context, size_t h_dim, size_t w_dim)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;
    out_shape->SetDim(h_dim, 1);
    out_shape->SetDim(w_dim, 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingCalculated(
    gert::InferShapeContext* context, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %zu must be 4!", ksize->GetSize()), return GRAPH_FAILED);
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        pads->GetSize() != PAD_SIZE, OP_LOGE(context->GetNodeName(), "Length of pads %zu must be 4!", pads->GetSize()),
        return GRAPH_FAILED);
    auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 4!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %ld and w %ld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[h_dim], strides_data[w_dim]),
        return GRAPH_FAILED);
    auto pads_data = static_cast<const int64_t*>(pads->GetData());

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;
    int64_t dim_size = in_shape->GetDim(h_dim);
    dim_size = CalculateUpdateDim(
        ksize_data[h_dim], pads_data[PAD_TOP], pads_data[PAD_BOTTOM], strides_data[h_dim], *ceil_mode, dim_size);
    out_shape->SetDim(h_dim, dim_size);
    dim_size = in_shape->GetDim(w_dim);
    dim_size = CalculateUpdateDim(
        ksize_data[w_dim], pads_data[PAD_LEFT], pads_data[PAD_RIGHT], strides_data[w_dim], *ceil_mode, dim_size);
    out_shape->SetDim(w_dim, dim_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingValid(
    gert::InferShapeContext* context, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %zu must be 4!", ksize->GetSize()), return GRAPH_FAILED);
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 4!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %ld and w %ld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[h_dim], strides_data[w_dim]),
        return GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(h_dim);
    out_shape->SetDim(h_dim, SameUpdateDim(ksize_data[h_dim], strides_data[h_dim], dim_size));
    dim_size = in_shape->GetDim(w_dim);
    out_shape->SetDim(w_dim, SameUpdateDim(ksize_data[w_dim], strides_data[w_dim], dim_size));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(
    gert::InferShapeContext* context, size_t h_dim, size_t w_dim, const gert::RuntimeAttrs* attrs)
{
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != SHAPE_NHWC_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %zu must be 4!", strides->GetSize()), return GRAPH_FAILED);
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    OP_CHECK_IF(
        strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0,
        OP_LOGE(
            context->GetNodeName(), "%s h %ld and w %ld must be greater than 0.",
            Int64ToString(strides_data, strides->GetSize()).c_str(), strides_data[h_dim], strides_data[w_dim]),
        return GRAPH_FAILED);

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(h_dim);
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

static ge::graphStatus InferShape4MaxPoolV3(gert::InferShapeContext* context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    if (Ops::Base::IsUnknownRank(*in_shape)) {
        Ops::Base::SetUnknownShape(INSHAPE_DIM_NUM, *out_shape);
        return ge::GRAPH_SUCCESS;
    }

    auto src_td = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, src_td);
    auto input_format = src_td->GetOriginFormat();
    size_t h_dim = input_format == FORMAT_NHWC ? 1 : 2;
    size_t w_dim = input_format == FORMAT_NHWC ? 2 : 3;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* global_pooling = attrs->GetAttrPointer<bool>(INDEX_GLOBAL_POOLING);
    OP_CHECK_NULL_WITH_CONTEXT(context, global_pooling);
    if (*global_pooling) {
        return InferShapeGlobalPooling(context, h_dim, w_dim);
    }

    auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING_MODE);
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
    return it->second(context, h_dim, w_dim, attrs);
}

static ge::graphStatus InferDtype4MaxPoolV3(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "MaxPoolV3InferDtype enter");
    // Get input tout
    auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);

    OP_LOGD(context->GetNodeName(), "MaxPoolV3InferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolV3).InferShape(InferShape4MaxPoolV3).InferDataType(InferDtype4MaxPoolV3);
} // namespace ops
