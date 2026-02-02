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
 * \file max_pool3d_grad_with_argmax_infershape_arch35.cpp
 * \brief
 */
#include <string>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t ATTR_INDEX_KSIZE = 0;
static constexpr size_t ATTR_INDEX_STRIDES = 1;
static constexpr size_t ATTR_INDEX_PADS = 2;
static constexpr size_t ATTR_INDEX_DILATION = 3;
static constexpr size_t ATTR_INDEX_CEIL_MODE = 4;
static constexpr size_t ATTR_INDEX_DATA_FORMAT = 5;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 3;
static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_ONE = 1;
static constexpr size_t INDEX_TWO = 2;
static constexpr size_t CDHW_DIM_NUM = 4;
static constexpr size_t NUMBER_TWO = 2;
static constexpr size_t PARAM_NUM = 4;
static constexpr size_t SHAPE_D_DIM = 2;
static constexpr size_t SHAPE_H_DIM = 3;
static constexpr size_t SHAPE_W_DIM = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static int64_t DivRtn(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE("MaxPool3DGradWithArgmax", "strides value cannot be zero.");
        return GRAPH_FAILED;
    }
    if (x < 0) {
        OP_LOGE("MaxPool3DGradWithArgmax", "x value cannot small than zero.");
        return GRAPH_FAILED;
    }
    int64_t q = x / y;
    return q;
}

static void UpdateMaxShape(
    const int64_t (&param)[PARAM_NUM], bool ceil_mode, const int64_t& dim_size, int64_t& out_max_shape)
{
    int64_t ksize = param[ATTR_INDEX_KSIZE];
    int64_t strides = param[ATTR_INDEX_STRIDES];
    int64_t pad = param[ATTR_INDEX_PADS];
    int64_t dilation = param[PARAM_NUM - 1];
    int64_t exact_size = dim_size + 2 * pad - dilation * (ksize - 1) - 1 + (ceil_mode ? (strides - 1) : 0);
    out_max_shape = DivRtn(exact_size, strides) + 1;
    if (ceil_mode) {
        if ((out_max_shape - 1) * strides >= dim_size + pad) {
            out_max_shape = out_max_shape - 1;
        }
    }
}

static int64_t CalcOutDim(int64_t input_dim, int64_t k, int64_t s, int64_t p, int64_t d, bool ceil_mode)
{
    int64_t param[PARAM_NUM] = {k, s, p, d};
    int64_t out_dim = 0;
    UpdateMaxShape(param, ceil_mode, input_dim, out_dim);
    return out_dim;
}

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPool3DGradWithArgmax(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGradWithArgmax infershape running");
    auto inputXDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXDesc);
    auto inputXFormat = inputXDesc->GetOriginFormat();
    OP_CHECK_IF(
        inputXFormat != FORMAT_ND && inputXFormat != FORMAT_NCDHW && inputXFormat != FORMAT_NDHWC && 
            inputXFormat != FORMAT_NCHW, 
        OP_LOGE(context->GetNodeName(), "format only supports ND, NCDHW, NDHWC"), return GRAPH_FAILED);

    size_t input_d_dim = SHAPE_D_DIM;
    size_t input_h_dim = SHAPE_H_DIM;
    size_t input_w_dim = SHAPE_W_DIM;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_IF(
        ksize->GetSize() != 1 && ksize->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of ksize %lu must be equal 1 or 3!", ksize->GetSize()),
        return GRAPH_FAILED);

    auto ksize_data = reinterpret_cast<const int64_t*>(ksize->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(ksize->GetSize()); i++) {
        OP_CHECK_IF(
            (ksize_data[i] <= 0),
            OP_LOGE(
                context->GetNodeName(), "Attr value invalid, ksize_data[%u] is %ld, should bigger than 0.", i,
                ksize_data[i]),
            return ge::GRAPH_FAILED);
    }

    int64_t kD = ksize_data[INDEX_ZERO];
    int64_t kH = ksize->GetSize() == 1 ? kD : ksize_data[INDEX_ONE];
    int64_t kW = ksize->GetSize() == 1 ? kD : ksize_data[INDEX_TWO];

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, strides);
    OP_CHECK_IF(
        strides->GetSize() != 0 && strides->GetSize() != 1 && strides->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of strides %lu must be equal 0 or 1 or 3!", strides->GetSize()),
        return GRAPH_FAILED);

    auto strides_data = reinterpret_cast<const int64_t*>(strides->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(strides->GetSize()); i++) {
        OP_CHECK_IF(
            (strides_data[i] <= 0),
            OP_LOGE(
                context->GetNodeName(), "Attr value invalid, strides_data[%u] is %ld, should bigger than 0.", i,
                strides_data[i]),
            return ge::GRAPH_FAILED);
    }

    int64_t sD = kD;
    int64_t sH = kH;
    int64_t sW = kW;
    if (strides->GetSize() > 0) {
        sD = strides_data[INDEX_ZERO];
        sH = (strides->GetSize() > 1) ? strides_data[INDEX_ONE] : sD;
        sW = (strides->GetSize() > 1) ? strides_data[INDEX_TWO] : sD;
    }

    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pads);
    OP_CHECK_IF(
        pads->GetSize() != 1 && pads->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of pads %lu must be equal 1 or 3!", pads->GetSize()),
        return GRAPH_FAILED);

    auto pads_data = reinterpret_cast<const int64_t*>(pads->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(pads->GetSize()); i++) {
        OP_CHECK_IF(
            (pads_data[i] < 0),
            OP_LOGE(
                context->GetNodeName(), "Attr value invalid, pads_data[%u] is %ld, should bigger or equal 0.", i,
                pads_data[i]),
            return ge::GRAPH_FAILED);
    }

    int64_t pD = pads_data[INDEX_ZERO];
    int64_t pH = pads->GetSize() == 1 ? pD : pads_data[INDEX_ONE];
    int64_t pW = pads->GetSize() == 1 ? pD : pads_data[INDEX_TWO];

    auto dilation = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_DILATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, dilation);
    OP_CHECK_IF(
        dilation->GetSize() != 1 && dilation->GetSize() != ATTR_LIST_SHAPE_SIZE,
        OP_LOGE(context->GetNodeName(), "Length of dilation %lu must be equal 1 or 3!", dilation->GetSize()),
        return GRAPH_FAILED);

    auto dilation_data = reinterpret_cast<const int64_t*>(dilation->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(dilation->GetSize()); i++) {
        OP_CHECK_IF(
            (dilation_data[i] <= 0),
            OP_LOGE(
                context->GetNodeName(), "Attr value invalid, dilation_data[%u] is %ld, should bigger than 0.", i,
                dilation_data[i]),
            return ge::GRAPH_FAILED);
    }

    int64_t dD = dilation_data[INDEX_ZERO];
    int64_t dH = dilation->GetSize() == 1 ? dD : dilation_data[INDEX_ONE];
    int64_t dW = dilation->GetSize() == 1 ? dD : dilation_data[INDEX_TWO];

    auto ceil_mode = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, ceil_mode);

    const char* data_format = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_format);

    OP_CHECK_IF(
        (pD > (kD / 2)) || (pH > (kH / 2)) || (pW > (kW / 2)),
        OP_LOGE(context->GetNodeName(), "Attr size invalid, padSize should smaller than kernelSize div 2"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (pD > ((kD - 1) * dD + 1) / 2) || (pH > ((kH - 1) * dH + 1) / 2) || (pW > ((kW - 1) * dW + 1) / 2),
        OP_LOGE(
            context->GetNodeName(),
            "Attr size invalid, padSize should smaller than ((kernelSize - 1) * dilation + 1) / 2."),
        return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* gradShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* argmaxShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, argmaxShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    size_t xDimNum = xShape->GetDimNum();

    std::string data_format_str = data_format;
    if (data_format_str == "NDHWC" || xDimNum == CDHW_DIM_NUM) {
        input_d_dim = input_d_dim - 1;
        input_h_dim = input_h_dim - 1;
        input_w_dim = input_w_dim - 1;
    }

    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*gradShape) ||
        Ops::Base::IsUnknownShape(*argmaxShape)) {
        SetAllUnknownDim(xDimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGradWithArgmax infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGradWithArgmax infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }
    yShape->SetDimNum(xDimNum);
    *yShape = *xShape;

    // Check gradShape and argmaxShape input dim invaild
    int64_t doExpected = CalcOutDim(xShape->GetDim(input_d_dim), kD, sD, pD, dD, *ceil_mode);
    int64_t hoExpected = CalcOutDim(xShape->GetDim(input_h_dim), kH, sH, pH, dH, *ceil_mode);
    int64_t woExpected = CalcOutDim(xShape->GetDim(input_w_dim), kW, sW, pW, dW, *ceil_mode);

    OP_CHECK_IF(
        (!Ops::Base::IsUnknownRank(*gradShape) && !Ops::Base::IsUnknownShape(*gradShape)) &&
            ((doExpected <= 0) || (doExpected != static_cast<int64_t>(gradShape->GetDim(input_d_dim))) ||
             (hoExpected <= 0) || (hoExpected != static_cast<int64_t>(gradShape->GetDim(input_h_dim))) ||
             (woExpected <= 0) || (woExpected != static_cast<int64_t>(gradShape->GetDim(input_w_dim)))),
        OP_LOGE(
            context->GetNodeName(), "GradShape D / H / W size invalid, expected d: %ld, h: %ld, w: %ld.", doExpected,
            hoExpected, woExpected),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (!Ops::Base::IsUnknownRank(*argmaxShape) && !Ops::Base::IsUnknownShape(*argmaxShape)) &&
            ((doExpected <= 0) || (doExpected != static_cast<int64_t>(argmaxShape->GetDim(input_d_dim))) ||
             (hoExpected <= 0) || (hoExpected != static_cast<int64_t>(argmaxShape->GetDim(input_h_dim))) ||
             (woExpected <= 0) || (woExpected != static_cast<int64_t>(argmaxShape->GetDim(input_w_dim)))),
        OP_LOGE(
            context->GetNodeName(), "ArgmaxShape D / H / W size invalid, expected d: %ld, h: %ld, w: %ld.", doExpected,
            hoExpected, woExpected),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGradWithArgmax infershape run success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPool3DGradWithArgmax(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(INDEX_OUT_MAX, xDtype);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3DGradWithArgmax)
    .InferShape(InferShapeForMaxPool3DGradWithArgmax)
    .InferDataType(InferDataTypeForMaxPool3DGradWithArgmax);
} // namespace ops