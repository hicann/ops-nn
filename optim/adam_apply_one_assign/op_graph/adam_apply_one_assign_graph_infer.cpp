/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adam_apply_one_assign_graph_infer.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIndexM = 1U;
constexpr size_t kInputIndexV = 2U;
constexpr size_t kInputIndexInput3 = 3U;
constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kOutputIndex1 = 1U;
constexpr size_t kOutputIndex2 = 2U;
} // namespace

// 与rt1.0一致：out0/out1/out2 分别取输入m/v/input3（第1/2/3个输入）的dtype
static ge::graphStatus InferDataType4AdamApplyOneAssign(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferDtype***");
    context->SetOutputDataType(kOutputIndex0, context->GetInputDataType(kInputIndexM));
    context->SetOutputDataType(kOutputIndex1, context->GetInputDataType(kInputIndexV));
    context->SetOutputDataType(kOutputIndex2, context->GetInputDataType(kInputIndexInput3));
    return GRAPH_SUCCESS;
}

IMPL_OP(AdamApplyOneAssign).InferDataType(InferDataType4AdamApplyOneAssign);

} // namespace ops
