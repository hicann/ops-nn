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
 * \file single_layer_lstm_infershape.cpp
 * \brief SingleLayerLstm InferShape / InferDataType.
 *
 * All eight outputs are [T, B, H]: T and B come from x, H from init_h. H is read off init_h rather
 * than divided out of w's second dimension, because w's shape is CHECKED against I and H in tiling
 * -- inferring H from w would make the same number arrive from two directions and one of them could
 * be wrong without contradiction.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t IDX_IN_X = 0;
constexpr size_t IDX_IN_W = 1;
constexpr size_t IDX_IN_INIT_H = 3;
constexpr size_t OUT_COUNT = 8;
constexpr size_t X_RANK = 3;
constexpr size_t STATE_RANK = 2;
constexpr size_t DIM_T = 0;
constexpr size_t DIM_B = 1;
constexpr size_t DIM_H = 1;
} // namespace

static ge::graphStatus InferShapeSingleLayerLstm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSingleLayerLstm");

    const gert::Shape* xShape = context->GetInputShape(IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* initHShape = context->GetInputShape(IDX_IN_INIT_H);
    OP_CHECK_NULL_WITH_CONTEXT(context, initHShape);

    if (xShape->GetDimNum() != X_RANK || initHShape->GetDimNum() != STATE_RANK) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "expected x rank 3 and init_h rank 2; got %zu and %zu",
                               xShape->GetDimNum(), initHShape->GetDimNum());
        return GRAPH_FAILED;
    }

    const int64_t timeStep = xShape->GetDim(DIM_T);
    const int64_t batch = xShape->GetDim(DIM_B);
    const int64_t hidden = initHShape->GetDim(DIM_H);

    for (size_t k = 0; k < OUT_COUNT; ++k) {
        gert::Shape* outShape = context->GetOutputShape(k);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        outShape->SetDimNum(X_RANK);
        outShape->SetDim(0, timeStep);
        outShape->SetDim(1, batch);
        outShape->SetDim(2, hidden);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSingleLayerLstm");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSingleLayerLstm(gert::InferDataTypeContext* context)
{
    for (size_t k = 0; k < OUT_COUNT; ++k) {
        context->SetOutputDataType(k, context->GetInputDataType(IDX_IN_W));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SingleLayerLstm).InferShape(InferShapeSingleLayerLstm).InferDataType(InferDataTypeSingleLayerLstm);

} // namespace ops
