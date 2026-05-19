/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hardshrink_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(HardShrinkGrad);

const aclTensor* HardShrinkGrad(
    const aclTensor* gradOutput, const aclTensor* self, float lambd, aclOpExecutor* executor)
{
    L0_DFX(HardShrinkGrad, gradOutput, self, lambd);
    op::Shape resultShape;
    BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), resultShape);
    auto result = executor->AllocTensor(resultShape, self->GetDataType(), op::Format::FORMAT_ND);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将HardShrinkGrad算子加入任务队列
    // HardShrinkGrad是算子的OpType，gradOutput, self是算子的输入，result是算子的输出
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(HardShrinkGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(result), OP_ATTR(lambd));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardShrinkGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);

    return result;
}
} // namespace l0op