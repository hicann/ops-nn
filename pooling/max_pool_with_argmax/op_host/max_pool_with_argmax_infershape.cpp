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
 * \file max_pool_with_argmax_infershape.cpp
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
namespace ops
{
static constexpr size_t INDEX_KSIZE = 0;
static constexpr size_t INDEX_STRIDES = 1;
static constexpr size_t INDEX_PADS = 2;
static constexpr size_t INDEX_DTYPE = 3;
static constexpr size_t INDEX_DATA_FORMAT = 5;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 4;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t INDEX_OUT_INDICES = 1;
static constexpr size_t PARAM_H_DIM = 0;
static constexpr size_t PARAM_W_DIM = 1;
static constexpr size_t SHAPE_H_DIM = 1;
static constexpr size_t SHAPE_W_DIM = 2;
static constexpr size_t INT32_DTYPE = 3;
static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_ONE = 1;
static constexpr size_t INDEX_TWO = 2;
static constexpr size_t INDEX_THREE = 3;
static constexpr size_t KSIZE_STRIDES_FIXED_DIM_VALUE = 1;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;


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

ge::graphStatus InferShapeForMaxPoolWithArgmax(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmax infershape running");
    auto src_td = context->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
    auto input_format = src_td->GetOriginFormat();
    auto indices_td = context->GetOutputDesc(INDEX_OUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indices_td);
    auto indices_dtype = indices_td->GetDataType();
    OP_LOGD(context->GetNodeName(), "indices_dtype = %d", indices_dtype);

    if (input_format != FORMAT_ND && input_format != FORMAT_NCHW && input_format != FORMAT_NHWC) {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
            Ops::Base::ToString(input_format).c_str(), "ND, NCHW or NHWC");
        return GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_KSIZE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ksize);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "ksize",
            std::to_string(ksize->GetSize()).c_str(), std::to_string(ATTR_LIST_SHAPE_SIZE).c_str());
        return GRAPH_FAILED;
    }
    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_STRIDES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, strides);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "strides",
            std::to_string(strides->GetSize()).c_str(), std::to_string(ATTR_LIST_SHAPE_SIZE).c_str());
        return GRAPH_FAILED;
    }
    auto strides_data = static_cast<const int64_t*>(strides->GetData());

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    OP_LOGE_IF(dataFormatPtr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get dataFormat failed.");
    std::string dataFormatStr(dataFormatPtr);
    if (ksize_data[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE &&
        strides_data[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
        std::string attrMsg = std::to_string(ksize_data[INDEX_ZERO]) + " and " +
                              std::to_string(strides_data[INDEX_ZERO]);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "ksize[0] and strides[0]", attrMsg.c_str(),
            "ksize[0] and strides[0] should be 1");
        return GRAPH_FAILED;
    }
    if (dataFormatStr == "NHWC") {
        if (ksize_data[INDEX_THREE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize[3]",
                std::to_string(ksize_data[INDEX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[INDEX_THREE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides[3]",
                std::to_string(strides_data[INDEX_THREE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    } else {
        // NCHW
        if (ksize_data[INDEX_ONE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ksize[1]",
                std::to_string(ksize_data[INDEX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
        if (strides_data[INDEX_ONE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "strides[1]",
                std::to_string(strides_data[INDEX_ONE]).c_str(), "1");
            return GRAPH_FAILED;
        }
    }

    const char* paddingPtr = attrs->GetAttrPointer<char>(INDEX_PADS);
    OP_LOGE_IF(paddingPtr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get pads failed.");
    std::string padStr(paddingPtr);
    if (padStr != "SAME" && padStr != "VALID") {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding", padStr.c_str(), "SAME or VALID");
        return GRAPH_FAILED;
    }

    const gert::Shape* in_shape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    gert::Shape* out_shape = context->GetOutputShape(INDEX_OUT_MAX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    gert::Shape* out_indices_shape = context->GetOutputShape(INDEX_OUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_indices_shape);
    *out_indices_shape = *in_shape;
    auto dimNums = in_shape->GetDimNum();
    if (Ops::Base::IsUnknownShape(*in_shape)) {
        SetAllUnknownDim(dimNums, out_shape);
        SetAllUnknownDim(dimNums, out_indices_shape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmax infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*in_shape)) {
        Ops::Base::SetUnknownRank(*out_shape);
        Ops::Base::SetUnknownRank(*out_indices_shape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmax infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    int h_dim = SHAPE_H_DIM, w_dim = SHAPE_W_DIM;
    if (dataFormatStr == "NCHW") {
        // NCHW
        h_dim = INDEX_TWO;
        w_dim = INDEX_THREE;
    }
    if (padStr == "SAME") {
        int64_t outH = (in_shape->GetDim(h_dim) + strides_data[h_dim] - 1) / strides_data[h_dim];
        int64_t outW = (in_shape->GetDim(w_dim) + strides_data[w_dim] - 1) / strides_data[w_dim];
        if (outH < 0 || outW < 0) {
            std::string valMsg = "[" + std::to_string(in_shape->GetDim(h_dim)) + ", " +
                                 std::to_string(in_shape->GetDim(w_dim)) + "], [" +
                                 std::to_string(strides_data[h_dim]) + ", " +
                                 std::to_string(strides_data[w_dim]) + "]";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "x[h_dim, w_dim], strides[h_dim, w_dim]", valMsg.c_str(),
                "these input values lead to inferred output H and W dimensions less than 0");
            return GRAPH_FAILED;
        }
        out_shape->SetDim(h_dim, outH);
        out_shape->SetDim(w_dim, outW);
        out_indices_shape->SetDim(h_dim, outH);
        out_indices_shape->SetDim(w_dim, outW);
    } else {
        int64_t outH = (in_shape->GetDim(h_dim) - ksize_data[h_dim] + strides_data[h_dim]) / strides_data[h_dim];
        int64_t outW = (in_shape->GetDim(w_dim) - ksize_data[w_dim] + strides_data[w_dim]) / strides_data[w_dim];
        if (outH < 0 || outW < 0) {
            std::string valMsg = "[" + std::to_string(in_shape->GetDim(h_dim)) + ", " +
                                 std::to_string(in_shape->GetDim(w_dim)) + "], [" +
                                 std::to_string(ksize_data[h_dim]) + ", " +
                                 std::to_string(ksize_data[w_dim]) + "], [" +
                                 std::to_string(strides_data[h_dim]) + ", " +
                                 std::to_string(strides_data[w_dim]) + "]";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "x[h_dim, w_dim], kernel_size[h_dim, w_dim], strides[h_dim, w_dim]", valMsg.c_str(),
                "these input values lead to inferred output H and W dimensions less than 0");
            return GRAPH_FAILED;
        }
        out_shape->SetDim(h_dim, outH);
        out_shape->SetDim(w_dim, outW);
        out_indices_shape->SetDim(h_dim, outH);
        out_indices_shape->SetDim(w_dim, outW);
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPoolWithArgmax infershape run success.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMaxPoolWithArgmax(gert::InferDataTypeContext* context)
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

IMPL_OP_INFERSHAPE(MaxPoolWithArgmax)
    .InferShape(InferShapeForMaxPoolWithArgmax)
    .InferDataType(InferDataTypeForMaxPoolWithArgmax);
}  // namespace ops