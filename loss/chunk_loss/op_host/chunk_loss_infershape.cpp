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
static ge::graphStatus InferShapeChunkLoss(gert::InferShapeContext* context)
{
    auto hidden = context->GetInputShape(0);
    auto weight = context->GetInputShape(1);
    auto labels = context->GetInputShape(2);
    auto loss = context->GetOutputShape(0);
    auto maxOut = context->GetOutputShape(1);
    auto sumOut = context->GetOutputShape(2);
    auto dH = context->GetOutputShape(3);
    if (hidden == nullptr || weight == nullptr || labels == nullptr || loss == nullptr || maxOut == nullptr ||
        sumOut == nullptr || dH == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (hidden->GetDimNum() != 2 || weight->GetDimNum() != 2 || labels->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (hidden->GetDim(0) != labels->GetDim(0) || hidden->GetDim(1) != weight->GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    if (hidden->GetDim(0) < 1 || hidden->GetDim(1) < 1 || weight->GetDim(0) < 1) {
        return ge::GRAPH_FAILED;
    }
    int64_t m = hidden->GetDim(0);
    int64_t h = hidden->GetDim(1);
    loss->SetDimNum(0);
    maxOut->SetDimNum(1);
    maxOut->SetDim(0, m);
    sumOut->SetDimNum(1);
    sumOut->SetDim(0, m);
    dH->SetDimNum(2);
    dH->SetDim(0, m);
    dH->SetDim(1, h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkLoss(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkLoss).InferShape(InferShapeChunkLoss).InferDataType(InferDataTypeChunkLoss);
} // namespace ops
