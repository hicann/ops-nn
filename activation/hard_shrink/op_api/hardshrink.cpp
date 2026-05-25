/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hardshrink.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(HardShrink);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    // 只需要判断dtype
    return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* HardShrinkAiCore(
    const aclTensor* input, float lambd, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(HardShrinkAiCore, input, lambd, output);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore HardShrink算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HardShrink, OP_INPUT(input), OP_OUTPUT(output), OP_ATTR(lambd));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardShrinkAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}

const aclTensor* HardShrink(const aclTensor* input, float lambd, aclOpExecutor* executor)
{
    auto output = executor->AllocTensor(input->GetViewShape(), input->GetDataType());

    if (IsAiCoreSupport(input->GetDataType())) {
        return HardShrinkAiCore(input, lambd, output, executor);
    }

    return nullptr;
}
} // namespace l0op
