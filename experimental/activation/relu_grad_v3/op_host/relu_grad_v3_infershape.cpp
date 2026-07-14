/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file relu_grad_v3_infershape.cpp
 * \brief ReluGradV3算子的shape推理和数据类型推理实现
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include <algorithm>

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferShapeReluGradV3(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeReluGradV3");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* yShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* zShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);

    size_t xDimNum = xShape->IsScalar() ? 1 : xShape->GetDimNum();
    size_t yDimNum = yShape->IsScalar() ? 1 : yShape->GetDimNum();
    size_t outDimNum = std::max(xDimNum, yDimNum);
    zShape->SetDimNum(outDimNum);

    for (size_t i = 0; i < outDimNum; ++i) {
        int64_t xDim = 1;
        int64_t yDim = 1;
        if (i >= outDimNum - xDimNum) {
            xDim = xShape->IsScalar() ? 1 : xShape->GetDim(i - (outDimNum - xDimNum));
        }
        if (i >= outDimNum - yDimNum) {
            yDim = yShape->IsScalar() ? 1 : yShape->GetDim(i - (outDimNum - yDimNum));
        }
        OP_CHECK_IF(xDim != yDim && xDim != 1 && yDim != 1, OP_LOGE(context, "input shapes are not broadcastable"),
                    return ge::GRAPH_FAILED);
        zShape->SetDim(i, std::max(xDim, yDim));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeReluGradV3");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeReluGradV3(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeReluGradV3");

    ge::DataType dtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeReluGradV3");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReluGradV3).InferShape(InferShapeReluGradV3).InferDataType(InferDataTypeReluGradV3);
} // namespace ops
