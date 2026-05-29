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
 * \file fake_quant_with_min_max_args_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t OUTPUT_IDX_Y = 0;

static graphStatus InferShapeForFakeQuantWithMinMaxArgs(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFakeQuantWithMinMaxArgs");
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // y has same shape as x.
    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShapeForFakeQuantWithMinMaxArgs");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFakeQuantWithMinMaxArgs(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForFakeQuantWithMinMaxArgs");
    context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxArgs)
    .InferShape(InferShapeForFakeQuantWithMinMaxArgs)
    .InferDataType(InferDataTypeForFakeQuantWithMinMaxArgs);
} // namespace ops
