/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_normalization_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_DY = 1;
static constexpr int64_t IDX_GAMMA = 2;
static constexpr int64_t IDX_MEAN = 3;
static constexpr int64_t IDX_RSTD = 4;

static ge::graphStatus InferShapeGroupNormalizationGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGroupNormalizationGrad");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // dy、gamma 需与 x 完全相同
    for (auto idx : {IDX_DY, IDX_GAMMA}) {
        const gert::Shape* s = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, s);
        OP_CHECK_IF(s->GetDimNum() != xShape->GetDimNum(), OP_LOGE(context, "input %ld dimNum mismatch with x", idx),
                    return ge::GRAPH_FAILED);
        for (size_t d = 0; d < xShape->GetDimNum(); ++d) {
            OP_CHECK_IF(s->GetDim(d) != xShape->GetDim(d),
                        OP_LOGE(context, "input %ld dim %zu mismatch with x", idx, d), return ge::GRAPH_FAILED);
        }
    }
    // mean、rstd 需为 [x.dim0, x.dim1]（即 [N, G]）
    for (auto idx : {IDX_MEAN, IDX_RSTD}) {
        const gert::Shape* s = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, s);
        OP_CHECK_IF(s->GetDimNum() != 2 || s->GetDim(0) != xShape->GetDim(0) || s->GetDim(1) != xShape->GetDim(1),
                    OP_LOGE(context, "input %ld shape must be [N, G] matching x", idx), return ge::GRAPH_FAILED);
    }

    gert::Shape* dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    *dxShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeGroupNormalizationGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGroupNormalizationGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, inputDataType);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormalizationGrad)
    .InferShape(InferShapeGroupNormalizationGrad)
    .InferDataType(InferDataTypeGroupNormalizationGrad);
} // namespace ops
