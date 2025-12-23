/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file leakyrelu_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeLeakyreluV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLeakyreluV2");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 填充输出shape大小
    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLeakyreluV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LeakyreluV2).InferShape(InferShapeLeakyreluV2);
} // namespace ops