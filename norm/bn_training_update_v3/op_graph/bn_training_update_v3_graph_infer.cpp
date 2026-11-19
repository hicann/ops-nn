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
 * \file bn_training_update_v3_graph_infer.cpp
 * \brief BNTrainingUpdateV3 graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
namespace ops {
static constexpr int64_t IDX_IN_0 = 0;
static constexpr int64_t IDX_IN_3 = 3;
static constexpr int64_t IDX_OUT_0 = 0;
static constexpr int64_t IDX_OUT_1 = 1;
static constexpr int64_t IDX_OUT_2 = 2;
static constexpr int64_t IDX_OUT_3 = 3;
static constexpr int64_t IDX_OUT_4 = 4;

ge::graphStatus InferDataTypeBNTrainingUpdateV3(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBNTrainingUpdateV3");
    context->SetOutputDataType(IDX_OUT_0, context->GetRequiredInputDataType(IDX_IN_0));
    context->SetOutputDataType(IDX_OUT_1, context->GetRequiredInputDataType(IDX_IN_3));
    context->SetOutputDataType(IDX_OUT_2, context->GetRequiredInputDataType(IDX_IN_3));
    context->SetOutputDataType(IDX_OUT_3, context->GetRequiredInputDataType(IDX_IN_3));
    context->SetOutputDataType(IDX_OUT_4, context->GetRequiredInputDataType(IDX_IN_3));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeBNTrainingUpdateV3");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BNTrainingUpdateV3).InferDataType(InferDataTypeBNTrainingUpdateV3);
} // namespace ops
