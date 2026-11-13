/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sync_bn_training_update_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4SyncBNTrainingUpdate(gert::InferShapeContext* context)
{
    auto in_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SyncBNTrainingUpdate(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(1));
}

IMPL_OP_INFERSHAPE(SyncBNTrainingUpdate)
    .InferShape(InferShape4SyncBNTrainingUpdate)
    .InferDataType(InferDataType4SyncBNTrainingUpdate);
} // namespace ops
