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
 * \file max_pool_with_argmax_v3_infershape.cpp
 * \brief
 */
#include <string>
#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INDEX_KSIZE = 0;
static constexpr size_t INDEX_STRIDES = 1;
static constexpr size_t INDEX_PADS = 2;
static constexpr size_t INDEX_DILATION = 4;
static constexpr size_t INDEX_CEIL_MODE = 5;
static constexpr size_t INDEX_DATA_FORMAT = 6;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 2;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t INDEX_OUT_INDICES = 1;
static constexpr size_t C0_SIZE = 16;
static constexpr size_t PARAM_NUM = 4;
static constexpr size_t PARAM_H_DIM = 0;
static constexpr size_t PARAM_W_DIM = 1;
static constexpr size_t SHAPE_H_DIM = 2;
static constexpr size_t SHAPE_W_DIM = 3;
static constexpr size_t CHW_DIM = 3;

static int64_t DivRtn(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MaxPoolWithArgmaxV3", "y", std::to_string(y).c_str(),
                                              "Divisor y must not be zero");
        return GRAPH_FAILED;
    }
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    }
    return q;
}

static void UpdateMaxShape(const int64_t (&param)[PARAM_NUM], bool ceil_mode, const int64_t& dim_size,
                           int64_t& out_max_shape)
{
    int64_t ksize = param[INDEX_KSIZE];
    int64_t strides = param[INDEX_STRIDES];
    int64_t pad = param[INDEX_PADS];
    int64_t dilation = param[PARAM_NUM - 1];
    int64_t exact_size = dim_size + 2 * pad - dilation * (ksize - 1) - 1 + (ceil_mode ? (strides - 1) : 0);
    out_max_shape = DivRtn(exact_size, strides) + 1;
    if (ceil_mode) {
        if ((out_max_shape - 1) * strides >= dim_size + pad) {
            out_max_shape = out_max_shape - 1;
        }
    }
}

ge::graphStatus InferShapeForMaxPoolWithArgmaxV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmaxV3 infershape running");
    auto src_td = context->GetInputDesc(0);
    OP_CHECK_IF(
        src_td == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto input_format = src_td->GetOriginFormat();
    auto indices_td = context->GetOutputDesc(INDEX_OUT_INDICES);
    OP_CHECK_IF(indices_td == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "indices_td is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    auto indices_dtype = indices_td->GetDataType();
    OP_LOGD(context->GetNodeName(), "indices_dtype = %d", indices_dtype);

    if (input_format != FORMAT_ND && input_format != FORMAT_NCHW && input_format != FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(input_format).c_str(),
                                   "ND, NCHW or NHWC");
        return GRAPH_FAILED;
    }

    size_t param_h_dim = PARAM_H_DIM;
    size_t param_w_dim = PARAM_W_DIM;
    size_t input_h_dim = SHAPE_H_DIM;
    size_t input_w_dim = SHAPE_W_DIM;

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "ksize", std::to_string(ksize->GetSize()).c_str(), "2");
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "strides", std::to_string(strides->GetSize()).c_str(),
                                     "2");
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OP_CHECK_IF(
        pads == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "pads",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (pads->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "pads", std::to_string(pads->GetSize()).c_str(), "2");
        return GRAPH_FAILED;
    }
    auto pads_data = static_cast<const int64_t*>(pads->GetData());

    auto dilation = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_DILATION);
    OP_CHECK_IF(
        dilation == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "dilation",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (dilation->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "dilation", std::to_string(dilation->GetSize()).c_str(),
                                     "2");
        return GRAPH_FAILED;
    }
    auto dilation_data = static_cast<const int64_t*>(dilation->GetData());

    auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    OP_CHECK_IF(
        ceil_mode == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ceil_mode",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    const char* data_format = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    OP_CHECK_IF(
        data_format == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "data_format",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    std::string data_format_str = data_format;
    if (data_format_str == "NHWC") {
        input_h_dim = input_h_dim - 1UL;
        input_w_dim = input_w_dim - 1UL;
    }

    const gert::Shape* in_shape = context->GetInputShape(0);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    gert::Shape* out_max_shape = context->GetOutputShape(INDEX_OUT_MAX);
    OP_CHECK_IF(
        out_max_shape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "out_max_shape is null; InferShape cannot continue"})),
        return ge::GRAPH_FAILED);
    *out_max_shape = *in_shape;
    gert::Shape* out_indices_shape = context->GetOutputShape(INDEX_OUT_INDICES);
    OP_CHECK_IF(
        out_indices_shape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "out_indices_shape is null; InferShape cannot continue"})),
        return ge::GRAPH_FAILED);
    *out_indices_shape = *in_shape;

    if (Ops::Base::IsUnknownRank(*in_shape) || Ops::Base::IsUnknownShape(*in_shape)) {
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmaxV3 infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }

    size_t dim_num = in_shape->GetDimNum();
    int64_t max_dim = 0;
    if (dim_num == CHW_DIM) {
        input_h_dim = input_h_dim - 1UL;
        input_w_dim = input_w_dim - 1UL;
    }
    for (size_t i = 0; i < dim_num; i++) {
        int64_t input_dim = in_shape->GetDim(i);
        if (i == input_h_dim) {
            int64_t param[PARAM_NUM] = {ksize_data[param_h_dim], strides_data[param_h_dim], pads_data[param_h_dim],
                                        dilation_data[param_h_dim]};
            UpdateMaxShape(param, *ceil_mode, input_dim, max_dim);
            out_max_shape->SetDim(i, max_dim);
            out_indices_shape->SetDim(i, max_dim);
        } else if (i == input_w_dim) {
            int64_t param[PARAM_NUM] = {ksize_data[param_w_dim], strides_data[param_w_dim], pads_data[param_w_dim],
                                        dilation_data[param_w_dim]};
            UpdateMaxShape(param, *ceil_mode, input_dim, max_dim);
            out_max_shape->SetDim(i, max_dim);
            out_indices_shape->SetDim(i, max_dim);
        } else {
            out_max_shape->SetDim(i, input_dim);
            out_indices_shape->SetDim(i, input_dim);
        }
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmaxV3 infershape run success.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolWithArgmaxV3).InferShape(InferShapeForMaxPoolWithArgmaxV3);
} // namespace ops
