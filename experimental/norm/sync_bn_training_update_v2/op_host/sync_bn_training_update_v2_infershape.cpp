/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file sync_bn_training_update_v2_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static ge::graphStatus InferShapeSyncBnTrainingUpdateV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSyncBnTrainingUpdateV2");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* x1Shape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* y1Shape = context->GetOutputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);

    // 填充输出shape大小
    auto xShapeSize = xShape->GetDimNum();
    auto x1ShapeSize = x1Shape->GetDimNum();
    auto x2ShapeSize = x2Shape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    y1Shape->SetDimNum(x1ShapeSize);
    for (size_t i = 0; i < x1ShapeSize; i++) {
        int64_t dim = x1Shape->GetDim(i);
        y1Shape->SetDim(i, dim);
    }
    y2Shape->SetDimNum(x2ShapeSize);
    for (size_t i = 0; i < x2ShapeSize; i++) {
        int64_t dim = x2Shape->GetDim(i);
        y2Shape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSyncBnTrainingUpdateV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBnTrainingUpdateV2).InferShape(InferShapeSyncBnTrainingUpdateV2);
} // namespace ops