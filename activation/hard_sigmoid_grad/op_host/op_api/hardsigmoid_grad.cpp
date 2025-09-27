/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hardsigmoid_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(HardSigmoidGrad);
const aclTensor* HardSigmoidGrad(
    const aclTensor* grads, const aclTensor* x, float alpha, float beta, aclOpExecutor* executor)
{
    L0_DFX(HardSigmoidGrad, grads, x, alpha, beta);
    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    CHECK_RET(y != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HardSigmoidGrad, OP_INPUT(grads, x), OP_OUTPUT(y), OP_ATTR(alpha, beta));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardSigmoidGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return y;
}
} // namespace l0op