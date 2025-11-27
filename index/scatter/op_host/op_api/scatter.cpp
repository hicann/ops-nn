/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "scatter.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(Scatter);

// AiCore的执行逻辑
aclTensor* ScatterAiCore(
    const aclTensor* data, const aclTensor* indices, const aclTensor* updates, int64_t axis, aclTensor* out,
    aclOpExecutor* executor)
{
    static const string reduce("update");
    L0_DFX(ScatterAiCore, data, indices, updates, out, reduce, axis);

    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(Scatter, OP_INPUT(data, indices, updates), OP_OUTPUT(out), OP_ATTR(reduce, axis));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ScatterAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* Scatter(
    const aclTensor* data, const aclTensor* indices, const aclTensor* updates, int64_t axis, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(data->GetViewShape(), data->GetDataType());
    ScatterAiCore(data, indices, updates, axis, out, executor);
    return out;
}
} // namespace l0op
