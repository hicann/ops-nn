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
 * \file in_infer_v2_graph_infer.cpp
 * \brief INInferV2 inferDataType：y dtype 同 x；batch_mean/batch_variance dtype 同 mean/variance
 *        （对齐 canndev built-in ELMTWISE_INFER_SHAPEANDTYPE 的 type 推导语义）
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferDataTypeForINInferV2(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataTypeForINInferV2");
    // optional 输入缺席时框架会压实输入存储，raw index 的 GetInputDataType(3)/(4) 会错位，
    // 必须用 IR 原型序访问器（与 infershape 的 GetOptionalInputShape 同理）；
    // mean/variance 缺席（语义必需）时返回 DT_UNDEFINED，由后续校验/tiling 拦截
    const ge::DataType xDataType = context->GetRequiredInputDataType(0);    // x
    const ge::DataType meanDataType = context->GetOptionalInputDataType(3); // mean
    const ge::DataType varDataType = context->GetOptionalInputDataType(4);  // variance
    context->SetOutputDataType(0, xDataType);                               // y 同 x
    context->SetOutputDataType(1, meanDataType);                            // batch_mean 同 mean
    context->SetOutputDataType(2, varDataType);                             // batch_variance 同 variance
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(INInferV2).InferDataType(InferDataTypeForINInferV2);
} // namespace ops
