/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_add_layer_norm_graph_infer.cpp
 * \brief Data type inference for QuantizeAddLayerNorm.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int X1_IDX = 0;
static constexpr int Y_IDX = 0;
static constexpr int X_IDX = 1;

ge::graphStatus InferDataType4QuantizeAddLayerNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4QuantizeAddLayerNorm");
    context->SetOutputDataType(Y_IDX, ge::DT_INT8);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4QuantizeAddLayerNorm");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantizeAddLayerNorm).InferDataType(InferDataType4QuantizeAddLayerNorm);
} // namespace ops
