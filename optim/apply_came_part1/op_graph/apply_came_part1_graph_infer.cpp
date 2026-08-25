/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_came_part1_graph_infer.cpp
 * \brief ApplyCamePart1 operator graph data type inference.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kSumGradROutputIndex = 0;
constexpr size_t kSumGradCOutputIndex = 1;
constexpr size_t kSumGradRCOutputIndex = 2;
} // namespace

static ge::graphStatus InferDataTypeApplyCamePart1(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kSumGradROutputIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumGradCOutputIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumGradRCOutputIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyCamePart1).InferDataType(InferDataTypeApplyCamePart1);
} // namespace ops
