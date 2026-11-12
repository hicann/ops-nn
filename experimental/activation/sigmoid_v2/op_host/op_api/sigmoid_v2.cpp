/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sigmoid_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SigmoidV2);

const aclTensor* SigmoidV2(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(SigmoidV2, self);
    auto* sigmoidV2Out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(sigmoidV2Out != nullptr, nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SigmoidV2, OP_INPUT(self), OP_OUTPUT(sigmoidV2Out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SigmoidV2 ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return sigmoidV2Out;
}

} // namespace l0op
