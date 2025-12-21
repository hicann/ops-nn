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
 * \file cross_v2.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace {
constexpr int64_t LOG_ALPHA_LEN = 3;
constexpr int64_t STEP_INFO_IDX = 1;
constexpr int64_t LABEL_INFO_IDX = 2;
constexpr int64_t LABEL_COE = 2;
constexpr int64_t NUM_ALIGN = 8;
} // namespace
using namespace ge;
namespace ops {
static ge::graphStatus InferShapeCrossV2(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *x1Shape;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeCrossV2(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CrossV2).InferShape(InferShapeCrossV2).InferDataType(InferDataTypeCrossV2);
} // namespace ops
