/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "modulate.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Modulate);

// AICore的执行逻辑
static const aclTensor* ModulateAiCore(
    const aclTensor* self, const aclTensor* scale, const aclTensor* shift, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ModulateAiCore, self, scale, shift, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Modulate, OP_INPUT(self, scale, shift), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

// 目前只有AICore
const aclTensor* Modulate(
    const aclTensor* self, const aclTensor* scale, const aclTensor* shift, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    CHECK_RET(out != nullptr, nullptr);
    return ModulateAiCore(self, scale, shift, out, executor);
}

} // namespace l0op