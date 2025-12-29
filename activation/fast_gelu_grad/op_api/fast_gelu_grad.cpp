/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fast_gelu_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(FastGeluGrad);

const aclTensor* FastGeluGrad(const aclTensor* gradOutput, const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(FastGeluGrad, gradOutput, self);
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    CHECK_RET(gradInput != nullptr, nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FastGeluGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(gradInput));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FastGeluGrad launch kernel failed.");
        return nullptr;
    }
    return gradInput;
}
} // namespace l0op