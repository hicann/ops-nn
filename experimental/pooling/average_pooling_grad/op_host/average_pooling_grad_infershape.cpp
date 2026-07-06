/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file average_pooling_grad_infershape.cpp
 * \brief AveragePoolingGrad shape and dtype inference.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t ORIG_SHAPE_IDX = 0;
static constexpr size_t GRAD_OUTPUT_IDX = 1;
static constexpr size_t OUTPUT_IDX = 0;
static constexpr size_t NCHW_DIMS = 4;

static ge::graphStatus InferShapeAveragePoolingGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeAveragePoolingGrad");

    const gert::Tensor* shapeTensor = context->GetInputTensor(ORIG_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    const int64_t* shapeValue = shapeTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeValue);
    OP_CHECK_IF(shapeTensor->GetOriginShape().GetShapeSize() != NCHW_DIMS,
        OP_LOGE(context->GetNodeName(), "orig_input_shape must have 4 elements"), return GRAPH_FAILED);

    gert::Shape* gradInputShape = context->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradInputShape);
    gradInputShape->SetDimNum(NCHW_DIMS);
    for (size_t i = 0; i < NCHW_DIMS; ++i) {
        gradInputShape->SetDim(i, shapeValue[i]);
    }

    OP_LOGD(context->GetNodeName(), "End InferShapeAveragePoolingGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAveragePoolingGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_IDX, context->GetInputDataType(GRAD_OUTPUT_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AveragePoolingGrad).InferShape(InferShapeAveragePoolingGrad).InferDataType(InferDataTypeAveragePoolingGrad);
} // namespace ops
