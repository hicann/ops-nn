/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chamfer_distance_grad.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
namespace {
constexpr int64_t INDEX_XYZ1 = 0;
constexpr int64_t INDEX_XYZ2 = 1;
constexpr int64_t INDEX_IDX1 = 2;
constexpr int64_t INDEX_IDX2 = 3;
constexpr int64_t INDEX_GRAD_DIST1 = 4;
constexpr int64_t INDEX_GRAD_DIST2 = 5;

constexpr int64_t INDEX_GRAD_XYZ1 = 0;
constexpr int64_t INDEX_GRAD_XYZ2 = 1;

constexpr int64_t ZERO_AXIS = 0;
constexpr int64_t ONE_AXIS = 1;
constexpr int64_t TWO_AXIS = 2;
constexpr int64_t LAST_DIM = 2;
} // namespace
using namespace ge;
namespace ops {
static ge::graphStatus InferDataType4ChamferDistanceGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataType4ChamferDistanceGrad start");
    auto input_xyz1_dtype = context->GetInputDataType(INDEX_XYZ1);
    auto input_xyz2_dtype = context->GetInputDataType(INDEX_XYZ2);
    context->SetOutputDataType(INDEX_GRAD_XYZ1, input_xyz1_dtype);
    context->SetOutputDataType(INDEX_GRAD_XYZ2, input_xyz2_dtype);
    OP_LOGD(context, "InferDataType4ChamferDistanceGrad end");
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferShape4ChamferDistanceGrad(gert::InferShapeContext* context)
{
    const gert::Shape* grad_dist1_shape = context->GetInputShape(INDEX_GRAD_DIST1);
    const gert::Shape* grad_dist2_shape = context->GetInputShape(INDEX_GRAD_DIST2);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_dist1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_dist2_shape);
    gert::Shape* grad_xyz1_shape = context->GetOutputShape(INDEX_GRAD_XYZ1);
    gert::Shape* grad_xyz2_shape = context->GetOutputShape(INDEX_GRAD_XYZ2);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_xyz1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_xyz2_shape);

    // Input_shape->Output_shape : (B, N) -> (B, N, 2)
    grad_xyz1_shape->AppendDim(grad_dist1_shape->GetDim(ZERO_AXIS));
    grad_xyz1_shape->AppendDim(grad_dist1_shape->GetDim(ONE_AXIS));
    grad_xyz1_shape->AppendDim(LAST_DIM);
    grad_xyz2_shape->AppendDim(grad_dist2_shape->GetDim(ZERO_AXIS));
    grad_xyz2_shape->AppendDim(grad_dist2_shape->GetDim(ONE_AXIS));
    grad_xyz2_shape->AppendDim(LAST_DIM);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChamferDistanceGrad)
    .InferShape(InferShape4ChamferDistanceGrad)
    .InferDataType(InferDataType4ChamferDistanceGrad);
} // namespace ops