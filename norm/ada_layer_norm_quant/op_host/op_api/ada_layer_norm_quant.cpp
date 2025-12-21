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
 * \file ada_layer_norm_quant.cpp
 * \brief
 */
#include "ada_layer_norm_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AdaLayerNormQuant);

const std::tuple<aclTensor*, aclTensor*> AdaLayerNormQuant(
    const aclTensor* x, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* smoothScalesOptional, float epsilon, const char* quantMode,
    aclOpExecutor* executor)
{
    L0_DFX(AdaLayerNormQuant, x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, epsilon, quantMode);

    op::Shape scaleShape;
    size_t dimNum = x->GetViewShape().GetDimNum();
    for (size_t i = 0; i < dimNum - 1; i++) {
        scaleShape.AppendDim(x->GetViewShape().GetDim(i));
    }
    auto out = executor->AllocTensor(x->GetViewShape(), DataType::DT_INT8, x->GetViewFormat());
    auto quantScale = executor->AllocTensor(scaleShape, DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AdaLayerNormQuant, OP_INPUT(x, scale, shift, weightOptional, biasOptional, smoothScalesOptional),
        OP_OUTPUT(out, quantScale), OP_ATTR(epsilon));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaLayerNormQuantAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }
    return std::tie(out, quantScale);
}
} // namespace l0op