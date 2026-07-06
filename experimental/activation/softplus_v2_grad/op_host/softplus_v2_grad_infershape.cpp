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
 * \file softplus_backward_infer.cpp
 * \brief
 */

#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeSoftplusV2Grad(gert::InferShapeContext* context)
{
    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_0);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);

    // 填充输出shape大小
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftplusV2Grad).InferShape(InferShapeSoftplusV2Grad);
} // namespace ops
