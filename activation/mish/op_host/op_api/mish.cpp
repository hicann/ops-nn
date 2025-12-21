/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "mish.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Mish);

// AICORE算子kernel
static const aclTensor *MishAiCore(const aclTensor *self, aclTensor *out, aclOpExecutor *executor) {
    L0_DFX(MishAiCore, self, out);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将Aicore Mish算子加入任务队列
    // Mish是算子的OpType，self是算子的输入，out是算子的输出
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Mish, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "Mish ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

const aclTensor *Mish(const aclTensor *self, aclOpExecutor *executor) {
    auto mishOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    return MishAiCore(self, mishOut, executor);
}
}  // namespace l0op
