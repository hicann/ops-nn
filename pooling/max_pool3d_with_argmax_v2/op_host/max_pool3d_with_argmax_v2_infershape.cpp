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
 * \file max_pool3d_with_argmax_v2_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include <string>
#include "common_dtype.h"

using namespace ge;

namespace ops
{
static constexpr size_t INDEX_KSIZE = 0;
static constexpr size_t INDEX_STRIDES = 1;
static constexpr size_t INDEX_PADS = 2;
static constexpr size_t INDEX_DILATION = 3;
static constexpr size_t INDEX_CEIL_MODE = 4;
static constexpr size_t INDEX_DATA_FORMAT = 5;
static constexpr size_t INDEX_DTYPE = 6;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 3;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t INDEX_OUT_INDICES = 1;
static constexpr size_t PARAM_NUM = 4;
static constexpr size_t PARAM_D_DIM = 0;
static constexpr size_t PARAM_H_DIM = 1;
static constexpr size_t PARAM_W_DIM = 2;
static constexpr size_t SHAPE_D_DIM = 2;
static constexpr size_t SHAPE_H_DIM = 3;
static constexpr size_t SHAPE_W_DIM = 4;
static constexpr size_t CDHW_DIM = 4;
static constexpr size_t INT32_DTYPE = 3;
static constexpr size_t INT64_DTYPE = 9;

static int64_t DivRtn(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE("MaxPool3DWithArgmaxV2", "strides value cannot be zero.");
        return GRAPH_FAILED;
    }
    if (x < 0) {
        OP_LOGE("MaxPool3DWithArgmaxV2", "x value cannot small than zero.");
        return GRAPH_FAILED;
    }
    int64_t q = x / y;
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

ge::graphStatus InferShape4MaxPool3DWithArgmaxV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DWithArgmaxV2 infershape running");
    auto src_td = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
    auto input_format = src_td->GetOriginFormat();
    auto indices_td = context->GetOutputDesc(INDEX_OUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indices_td);
    auto indices_dtype = indices_td->GetDataType();
    OP_LOGD(context->GetNodeName(), "indices_dtype = %d", indices_dtype);
    OP_CHECK_IF(input_format != FORMAT_ND && input_format != FORMAT_NCDHW && input_format != FORMAT_NDHWC,
             OP_LOGE(context->GetNodeName(), "format only supports ND, NCDHW, NDHWC"),
             return GRAPH_FAILED);

    size_t param_d_dim = PARAM_D_DIM;
    size_t param_h_dim = PARAM_H_DIM;
    size_t param_w_dim = PARAM_W_DIM;
    size_t input_d_dim = SHAPE_D_DIM;
    size_t input_h_dim = SHAPE_H_DIM;
    size_t input_w_dim = SHAPE_W_DIM;

    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ksize);
    std::string errMsg4Ksize = optiling::ConcatString("Length of ksize ", ksize->GetSize(), " must be 3!");
    OP_CHECK_IF(ksize->GetSize() != ATTR_LIST_SHAPE_SIZE,
             OP_LOGE(context->GetNodeName(), "%s", errMsg4Ksize.c_str()), return GRAPH_FAILED);
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
    std::string errMsg4Strides = optiling::ConcatString("Length of strides ", strides->GetSize(), " must be 3!");
    OP_CHECK_IF(strides->GetSize() != ATTR_LIST_SHAPE_SIZE,
             OP_LOGE(context->GetNodeName(), "%s", errMsg4Strides.c_str()), return GRAPH_FAILED);
    auto strides_data = static_cast<const int64_t*>(strides->GetData());

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_PADS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, pads);
    std::string errMsg4Pads = optiling::ConcatString("Length of pads ", pads->GetSize(), " must be 3!");
    OP_CHECK_IF(pads->GetSize() != ATTR_LIST_SHAPE_SIZE,
             OP_LOGE(context->GetNodeName(), "%s", errMsg4Pads.c_str()), return GRAPH_FAILED);
    auto pads_data = static_cast<const int64_t*>(pads->GetData());

    auto dilation = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_DILATION);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dilation);
    std::string errMsg4Dilation = optiling::ConcatString("Length of dilation ", dilation->GetSize(), " must be 3!");
    OP_CHECK_IF(dilation->GetSize() != ATTR_LIST_SHAPE_SIZE,
             OP_LOGE(context->GetNodeName(), "%s", errMsg4Dilation.c_str()), return GRAPH_FAILED);
    auto dilation_data = static_cast<const int64_t*>(dilation->GetData());

    auto ceil_mode = attrs->GetAttrPointer<bool>(INDEX_CEIL_MODE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);

    const char* data_format = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, data_format);

    std::string data_format_str = data_format;
    if (data_format_str == "NDHWC") {
        input_d_dim = input_d_dim - 1;
        input_h_dim = input_h_dim - 1;
        input_w_dim = input_w_dim - 1;
    }
    
    const gert::Shape* in_shape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    gert::Shape* out_max_shape = context->GetOutputShape(INDEX_OUT_MAX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_max_shape);
    *out_max_shape = *in_shape;
    gert::Shape* out_indices_shape = context->GetOutputShape(INDEX_OUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_indices_shape);
    *out_indices_shape = *in_shape;

    if (Ops::Base::IsUnknownRank(*in_shape) || Ops::Base::IsUnknownShape(*in_shape)) {
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DWithArgmaxV2 infershape handle unknown rank or shape.");
        return ge::GRAPH_SUCCESS;
    }

    size_t dim_num = in_shape->GetDimNum();
    int64_t max_dim = 0;
    if (dim_num == CDHW_DIM) {
        input_d_dim = input_d_dim - 1;
        input_h_dim = input_h_dim - 1;
        input_w_dim = input_w_dim - 1;
    }
    for (size_t i = 0; i < dim_num; i++) {
        int64_t input_dim = in_shape->GetDim(i);
        if (i == input_d_dim) {
            int64_t param[PARAM_NUM] = {ksize_data[param_d_dim], strides_data[param_d_dim], pads_data[param_d_dim],
                                        dilation_data[param_d_dim]};
            UpdateMaxShape(param, *ceil_mode, input_dim, max_dim);
            out_max_shape->SetDim(i, max_dim);
            out_indices_shape->SetDim(i, max_dim);
        } else if (i == input_h_dim) {
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
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DWithArgmaxV2 infershape run success.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MaxPool3DWithArgmaxV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType x = context->GetInputDataType(0);
    context->SetOutputDataType(INDEX_OUT_MAX, x);

    auto attrsPtr = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_DTYPE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType indicesDtype = *dstDtype == INT32_DTYPE ? ge::DT_INT32 : ge::DT_INT64;

    context->SetOutputDataType(INDEX_OUT_INDICES, indicesDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3DWithArgmaxV2)
    .InferShape(InferShape4MaxPool3DWithArgmaxV2)
    .InferDataType(InferDataType4MaxPool3DWithArgmaxV2);
} // namespace ops