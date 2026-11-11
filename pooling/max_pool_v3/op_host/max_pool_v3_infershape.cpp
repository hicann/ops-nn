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
#include <algorithm>

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
constexpr size_t DIM_N = 0;
constexpr int64_t MAX_KSIZE_HW_PRODUCT = 255LL;

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
static inline int64_t CalculateUpdateDim(const int64_t ksize, const int64_t padL, const int64_t padR,
                                         const int64_t stride, const bool ceil_mode, int64_t& dim_size)
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
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "out_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    *out_shape = *in_shape;
    out_shape->SetDim(h_dim, 1);
    out_shape->SetDim(w_dim, 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingCalculated(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                                   const gert::RuntimeAttrs* attrs)
{
    const char* opName_ = "MaxPoolV3";
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_IF(
        pads == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "pads",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (pads->GetSize() != PAD_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of pads", std::to_string(pads->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    OP_CHECK_IF(
        ceil_mode == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ceil_mode",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    if (strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "strides[h,w]",
            (std::to_string(strides_data[h_dim]) + ", " + std::to_string(strides_data[w_dim])).c_str(),
            "strides must be greater than 0");
        return GRAPH_FAILED;
    }
    auto pads_data = static_cast<const int64_t*>(pads->GetData());

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "out_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    *out_shape = *in_shape;
    int64_t dim_size = in_shape->GetDim(h_dim);
    dim_size = CalculateUpdateDim(ksize_data[h_dim], pads_data[PAD_TOP], pads_data[PAD_BOTTOM], strides_data[h_dim],
                                  *ceil_mode, dim_size);
    out_shape->SetDim(h_dim, dim_size);
    dim_size = in_shape->GetDim(w_dim);
    dim_size = CalculateUpdateDim(ksize_data[w_dim], pads_data[PAD_LEFT], pads_data[PAD_RIGHT], strides_data[w_dim],
                                  *ceil_mode, dim_size);
    out_shape->SetDim(w_dim, dim_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingValid(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                              const gert::RuntimeAttrs* attrs)
{
    const char* opName_ = "MaxPoolV3";
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    if (strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "strides[h,w]",
            (std::to_string(strides_data[h_dim]) + ", " + std::to_string(strides_data[w_dim])).c_str(),
            "strides must be greater than 0");
        return GRAPH_FAILED;
    }

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "out_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(h_dim);
    out_shape->SetDim(h_dim, SameUpdateDim(ksize_data[h_dim], strides_data[h_dim], dim_size));
    dim_size = in_shape->GetDim(w_dim);
    out_shape->SetDim(w_dim, SameUpdateDim(ksize_data[w_dim], strides_data[w_dim], dim_size));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(gert::InferShapeContext* context, size_t h_dim, size_t w_dim,
                                             const gert::RuntimeAttrs* attrs)
{
    const char* opName_ = "MaxPoolV3";
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    if (strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "strides[h,w]",
            (std::to_string(strides_data[h_dim]) + ", " + std::to_string(strides_data[w_dim])).c_str(),
            "strides must be greater than 0");
        return GRAPH_FAILED;
    }

    auto in_shape = context->GetInputShape(0);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "out_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    *out_shape = *in_shape;

    int64_t dim_size = in_shape->GetDim(h_dim);
    out_shape->SetDim(h_dim, SameUpdateDim(1, strides_data[h_dim], dim_size));
    dim_size = in_shape->GetDim(w_dim);
    out_shape->SetDim(w_dim, SameUpdateDim(1, strides_data[w_dim], dim_size));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKsizeConstraints(gert::InferShapeContext* context, const gert::RuntimeAttrs* attrs)
{
    const char* opName_ = "MaxPoolV3";
    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of ksize", std::to_string(ksize->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    const char* data_format_ptr = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    std::string data_format = data_format_ptr == nullptr ? "NCHW" : data_format_ptr;
    size_t cDim = data_format == "NHWC" ? 3 : 1;
    size_t hDim = data_format == "NHWC" ? 1 : 2;
    size_t wDim = data_format == "NHWC" ? 2 : 3;
    if (ksize_data[DIM_N] != 1 || ksize_data[cDim] != 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "ksize[N,C]",
            (std::to_string(ksize_data[DIM_N]) + ", " + std::to_string(ksize_data[cDim])).c_str(),
            "ksize of N and C dimensions should be 1");
        return GRAPH_FAILED;
    }
    if (ksize_data[hDim] <= 0 || ksize_data[wDim] <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "ksize[h,w]", (std::to_string(ksize_data[hDim]) + ", " + std::to_string(ksize_data[wDim])).c_str(),
            "ksize of H and W dimensions should be greater than 0");
        return GRAPH_FAILED;
    }
    if (ksize_data[hDim] * ksize_data[wDim] > MAX_KSIZE_HW_PRODUCT) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "ksize[h,w]", (std::to_string(ksize_data[hDim]) + ", " + std::to_string(ksize_data[wDim])).c_str(),
            "the product of ksize in H and W dimensions should be less than or equal to 255");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"CALCULATED", InferShapePaddingCalculated},
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

static ge::graphStatus InferShape4MaxPoolV3(gert::InferShapeContext* context)
{
    const char* opName_ = "MaxPoolV3";
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "out_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*in_shape)) {
        Ops::Base::SetUnknownShape(INSHAPE_DIM_NUM, *out_shape);
        return ge::GRAPH_SUCCESS;
    }

    auto src_td = context->GetInputDesc(0);
    OP_CHECK_IF(
        src_td == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto input_format = src_td->GetOriginFormat();
    size_t h_dim = input_format == FORMAT_NHWC ? 1 : 2;
    size_t w_dim = input_format == FORMAT_NHWC ? 2 : 3;

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const bool* global_pooling = attrs->GetAttrPointer<bool>(INDEX_GLOBAL_POOLING);
    OP_CHECK_IF(
        global_pooling == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "global_pooling",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (*global_pooling) {
        return InferShapeGlobalPooling(context, h_dim, w_dim);
    }

    auto padding_mode = attrs->GetAttrPointer<char>(INDEX_PADDING_MODE);
    OP_CHECK_IF(
        padding_mode == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "padding_mode",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto it = std::find_if(kFuncMap.begin(), kFuncMap.end(),
                           [&padding_mode](const std::pair<std::string, InferShapePaddingFunc>& item) -> bool {
                               return item.first == padding_mode;
                           });
    if (it == kFuncMap.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "padding_mode", padding_mode,
                                              "must be in (CALCULATED, VALID, SAME)");
        return GRAPH_FAILED;
    }

    if (CheckKsizeConstraints(context, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    OP_CHECK_IF(
        ceil_mode == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ceil_mode",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (*ceil_mode && it->first != "CALCULATED") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ceil_mode", "true",
                                              "ceil_mode only supports false when padding_mode is SAME or VALID");
        return GRAPH_FAILED;
    }

    // when padding_mode in (CALCULATED, VALID, SAME)
    return it->second(context, h_dim, w_dim, attrs);
}

IMPL_OP_INFERSHAPE(MaxPoolV3).InferShape(InferShape4MaxPoolV3);
} // namespace ops
