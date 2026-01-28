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
 * \file batch_norm_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr int64_t DY_INPUT_IDX = 0;
static constexpr int64_t X_INPUT_IDX = 1;
static constexpr int64_t WEIGHT_INPUT_IDX = 2;

static constexpr int64_t DX_OUTPUT_IDX = 0;
static constexpr int64_t DWEIGHT_OUTPUT_IDX = 1;
static constexpr int64_t DBIAS_OUTPUT_IDX = 2;
static constexpr int64_t RESERVER_SPACE4_OUTPUT_IDX = 3;
static constexpr int64_t RESERVER_SPACE5_OUTPUT_IDX = 4;

static constexpr int DIM_0 = 0;
static constexpr int DIM_1 = 1;
static constexpr int DIM_3 = 3;
static constexpr int DIM_4 = 4;
static constexpr int DIM_LEN_ONE = 1;

static ge::graphStatus BatchNormGradInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* dyShape = context->GetInputShape(DY_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    const gert::Shape* weightShape = context->GetInputShape(WEIGHT_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    gert::Shape* dxShape = context->GetOutputShape(DX_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    gert::Shape* dweightShape = context->GetOutputShape(DWEIGHT_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dweightShape);
    gert::Shape* dbiasShape = context->GetOutputShape(DBIAS_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbiasShape);
    gert::Shape* reserverSpace4Shape = context->GetOutputShape(RESERVER_SPACE4_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserverSpace4Shape);
    gert::Shape* reserverSpace5Shape = context->GetOutputShape(RESERVER_SPACE5_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserverSpace5Shape);

    *dxShape = *dyShape;
    *dweightShape = *weightShape;
    *dbiasShape = *weightShape;

    *reserverSpace4Shape = {0};
    *reserverSpace5Shape = {0};
    return GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormGradInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType dyDtype = context->GetInputDataType(DY_INPUT_IDX);
    const ge::DataType weightDtype = context->GetInputDataType(WEIGHT_INPUT_IDX);

    context->SetOutputDataType(DX_OUTPUT_IDX, dyDtype);
    context->SetOutputDataType(DWEIGHT_OUTPUT_IDX, weightDtype);
    context->SetOutputDataType(DBIAS_OUTPUT_IDX, weightDtype);
    context->SetOutputDataType(RESERVER_SPACE4_OUTPUT_IDX, weightDtype);
    context->SetOutputDataType(RESERVER_SPACE5_OUTPUT_IDX, weightDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchNormGrad).InferShape(BatchNormGradInferShape).InferDataType(BatchNormGradInferDataType);
} // namespace ops
