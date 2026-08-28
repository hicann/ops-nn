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
 * \file dynamic_rnn_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
constexpr int X_SHAPE_SIZE_LIMIT = 3;
constexpr int CONSTANT_FOUR = 4;
constexpr int CONSTANT_TWO = 2;
constexpr int RNN_OUTPUT_INDEX_C = 2;
constexpr int RNN_OUTPUT_INDEX_I = 3;
constexpr int RNN_OUTPUT_INDEX_J = 4;
constexpr int RNN_OUTPUT_INDEX_F = 5;
constexpr int RNN_OUTPUT_INDEX_O = 6;
constexpr int RNN_OUTPUT_INDEX_TANHC = 7;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

static ge::graphStatus GetInputShapes(const gert::InferShapeContext* context, const gert::Shape** xShape,
                                      const gert::Shape** wShape)
{
    *xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, *xShape);
    *wShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, *wShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetOutputShapes(gert::InferShapeContext* context, gert::Shape** yShape,
                                       gert::Shape** outputhShape, gert::Shape** outputcShape, gert::Shape** iShape,
                                       gert::Shape** jShape, gert::Shape** fShape, gert::Shape** oShape,
                                       gert::Shape** tanhcShape)
{
    *yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, *yShape);
    *outputhShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, *outputhShape);
    *outputcShape = context->GetOutputShape(RNN_OUTPUT_INDEX_C);
    OP_CHECK_NULL_WITH_CONTEXT(context, *outputcShape);
    *iShape = context->GetOutputShape(RNN_OUTPUT_INDEX_I);
    OP_CHECK_NULL_WITH_CONTEXT(context, *iShape);
    *jShape = context->GetOutputShape(RNN_OUTPUT_INDEX_J);
    OP_CHECK_NULL_WITH_CONTEXT(context, *jShape);
    *fShape = context->GetOutputShape(RNN_OUTPUT_INDEX_F);
    OP_CHECK_NULL_WITH_CONTEXT(context, *fShape);
    *oShape = context->GetOutputShape(RNN_OUTPUT_INDEX_O);
    OP_CHECK_NULL_WITH_CONTEXT(context, *oShape);
    *tanhcShape = context->GetOutputShape(RNN_OUTPUT_INDEX_TANHC);
    OP_CHECK_NULL_WITH_CONTEXT(context, *tanhcShape);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetHiddenSize(const gert::Shape* wShape)
{
    if (wShape->GetDim(1) == UNKNOWN_DIM_VALUE) {
        return UNKNOWN_DIM_VALUE;
    }
    return wShape->GetDim(1) / CONSTANT_FOUR;
}

static void SetOutputShapes(gert::Shape* yShape, gert::Shape* outputhShape, gert::Shape* outputcShape,
                            gert::Shape* iShape, gert::Shape* jShape, gert::Shape* fShape, gert::Shape* oShape,
                            gert::Shape* tanhcShape, int64_t num_step, int64_t batch_size, int64_t hidden_size,
                            bool isBidirectional)
{
    const int64_t time_step = isBidirectional ? CONSTANT_TWO * num_step : num_step;
    const int64_t out_hidden_size = isBidirectional ? CONSTANT_TWO * hidden_size : hidden_size;
    *yShape = {num_step, batch_size, out_hidden_size};
    *outputhShape = {num_step, batch_size, out_hidden_size};
    *outputcShape = {num_step, batch_size, out_hidden_size};
    *iShape = {time_step, batch_size, hidden_size};
    *jShape = {time_step, batch_size, hidden_size};
    *fShape = {time_step, batch_size, hidden_size};
    *oShape = {time_step, batch_size, hidden_size};
    *tanhcShape = {time_step, batch_size, hidden_size};
}

static ge::graphStatus InferShape4DynamicRNN(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4DynamicRNN start");
    const gert::Shape* xShape = nullptr;
    const gert::Shape* wShape = nullptr;
    if (GetInputShapes(context, &xShape, &wShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* yShape = nullptr;
    gert::Shape* outputhShape = nullptr;
    gert::Shape* outputcShape = nullptr;
    gert::Shape* iShape = nullptr;
    gert::Shape* jShape = nullptr;
    gert::Shape* fShape = nullptr;
    gert::Shape* oShape = nullptr;
    gert::Shape* tanhcShape = nullptr;
    if (GetOutputShapes(context, &yShape, &outputhShape, &outputcShape, &iShape, &jShape, &fShape, &oShape,
                        &tanhcShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (xShape->GetDimNum() != X_SHAPE_SIZE_LIMIT) {
        OP_LOGE(context->GetNodeName(), "The input x shape dim is not 3, please check!");
        return ge::GRAPH_FAILED;
    }

    const int64_t num_step = xShape->GetDim(0);
    const int64_t batch_size = xShape->GetDim(1);
    const int64_t hidden_size = GetHiddenSize(wShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* direction = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, direction);

    SetOutputShapes(yShape, outputhShape, outputcShape, iShape, jShape, fShape, oShape, tanhcShape, num_step,
                    batch_size, hidden_size, strcmp(direction, "BIDIRECTIONAL") == 0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DynamicRNN(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN start");
    auto input_x_dtype = context->GetInputDataType(0);
    auto input_b_dtype = context->GetInputDataType(CONSTANT_TWO);

    OP_CHECK_IF(context->SetOutputDataType(0, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType y Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(1, input_x_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType output_h Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_C, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType output_c Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_I, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType i Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_J, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType j Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_F, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType f Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_O, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType o Fail"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetOutputDataType(RNN_OUTPUT_INDEX_TANHC, input_b_dtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetOutputDataType tanhc Fail"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "InferDataType4DynamicRNN end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicRNN).InferShape(InferShape4DynamicRNN).InferDataType(InferDataType4DynamicRNN);
} // namespace ops
