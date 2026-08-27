/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kMInIndex = 1;
constexpr size_t kMOutIndex = 0;
constexpr size_t kSumURIndex = 1;
constexpr size_t kSumUCIndex = 2;
constexpr size_t kSumURCIndex = 3;
} // namespace

static ge::graphStatus InferDataTypeApplyCamePart3(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kMOutIndex, context->GetInputDataType(kMInIndex));
    context->SetOutputDataType(kSumURIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumUCIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kSumURCIndex, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyCamePart3).InferDataType(InferDataTypeApplyCamePart3);
} // namespace ops
