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
 * \file sgd_graph_infer.cpp
 * \brief SGD InferDataType：唯一图输出 parameters 的 dtype 等于输入 parameters。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t PARAMETERS_INDEX = 0;
} // namespace

static ge::graphStatus InferDataTypeForSgd(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataTypeForSgd begin.");
    // accum / stat 不是图输出（靠覆写输入 GM 原地回写），故此处只有一路。
    context->SetOutputDataType(PARAMETERS_INDEX, context->GetInputDataType(PARAMETERS_INDEX));
    OP_LOGD(context, "InferDataTypeForSgd end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SGD).InferDataType(InferDataTypeForSgd);
} // namespace ops
