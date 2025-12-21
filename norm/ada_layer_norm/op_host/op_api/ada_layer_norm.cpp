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
 * \file ada_layer_norm.cpp
 * \brief
 */
#include "ada_layer_norm.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AdaLayerNorm);

const aclTensor* AdaLayerNorm(AdaLayerNormInputTensor inputTensor, float epsilon, aclOpExecutor* executor)
{
    L0_DFX(AdaLayerNorm, inputTensor.x, inputTensor.scale, inputTensor.shift, inputTensor.weightOptional, inputTensor.biasOptional, epsilon);
    auto out = executor->AllocTensor(inputTensor.x->GetViewShape(), inputTensor.x->GetDataType(), inputTensor.x->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AdaLayerNorm, OP_INPUT(inputTensor.x, inputTensor.scale, inputTensor.shift, inputTensor.weightOptional, inputTensor.biasOptional), OP_OUTPUT(out), OP_ATTR(epsilon));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaLayerNormAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}
} // namespace l0op