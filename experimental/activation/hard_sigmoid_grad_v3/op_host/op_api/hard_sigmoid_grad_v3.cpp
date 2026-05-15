/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hard_sigmoid_grad_v3.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(HardSigmoidGradV3);

const aclTensor *HardSigmoidGradV3(const aclTensor *gradOutput, const aclTensor *self, aclOpExecutor *executor)
{
    L0_DFX(HardSigmoidGradV3, gradOutput, self);
    auto out = executor->AllocTensor(
        gradOutput->GetStorageShape(), gradOutput->GetDataType(), gradOutput->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HardSigmoidGradV3, OP_INPUT(gradOutput, self), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardSigmoidGradV3 ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

}  // namespace l0op
