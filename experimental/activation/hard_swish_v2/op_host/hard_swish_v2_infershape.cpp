/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file hard_swish_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeHardSwishV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeHardSwishV2");

    // get input shapes
    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    auto yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 填充输出shape大小
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeHardSwishV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HardSwishV2).InferShape(InferShapeHardSwishV2);
} // namespace ops