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
 * \file fake_quant_affine_cachemask_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t OUTPUT_IDX_MASK = 1;

static ge::graphStatus FakeQuantAffineCachemaskInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do FakeQuantAffineCachemaskInferShape");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* maskShape = context->GetOutputShape(OUTPUT_IDX_MASK);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    *yShape = *xShape;
    *maskShape = *xShape;

    OP_LOGD(context, "End to do FakeQuantAffineCachemaskInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus FakeQuantAffineCachemaskInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "FakeQuantAffineCachemaskInferDtype enter");
    auto inputDtype = context->GetInputDataType(INPUT_IDX_X);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());
    context->SetOutputDataType(OUTPUT_IDX_Y, inputDtype);
    context->SetOutputDataType(OUTPUT_IDX_MASK, DT_BOOL);
    OP_LOGD(context, "y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(OUTPUT_IDX_Y)).c_str());
    OP_LOGD(context, "mask dtype: %s", Ops::Base::ToString(context->GetOutputDataType(OUTPUT_IDX_MASK)).c_str());
    OP_LOGD(context, "FakeQuantAffineCachemaskInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantAffineCachemask)
    .InferShape(FakeQuantAffineCachemaskInferShape)
    .InferDataType(FakeQuantAffineCachemaskInferDtype);
} // namespace ops
