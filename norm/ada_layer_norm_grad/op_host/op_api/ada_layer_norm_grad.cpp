/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ada_layer_norm_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AdaLayerNormGrad);

const std::array<aclTensor*, GRAD_OUT_NUM> AdaLayerNormGrad(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* rstd, const aclTensor* mean,
    const aclTensor* scale, const aclTensor* shift, const aclTensor* weight, 
    const aclTensor* bias,  aclOpExecutor* executor)
{
    L0_DFX(AdaLayerNormGrad, gradOut, input, rstd, mean, scale, weight, bias);
    auto gradInputOut = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), Format::FORMAT_ND);
    auto gradScale = executor->AllocTensor(scale->GetViewShape(), scale->GetDataType(), Format::FORMAT_ND);
    auto gradShift = executor->AllocTensor(shift->GetViewShape(), shift->GetDataType(), Format::FORMAT_ND);
    auto gradWeight = executor->AllocTensor(weight->GetViewShape(), weight->GetDataType(), Format::FORMAT_ND);
    auto gradBias = executor->AllocTensor(bias->GetViewShape(), bias->GetDataType(), Format::FORMAT_ND);


    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AdaLayerNormGrad, OP_INPUT(gradOut, input, rstd, mean, scale, weight, bias), OP_OUTPUT(gradInputOut,  gradScale, gradShift, gradWeight, gradBias));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaLayerNormGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::array<aclTensor*, GRAD_OUT_NUM>{nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    return {gradInputOut, gradScale, gradShift, gradWeight, gradBias};
}

} // namespace l0op