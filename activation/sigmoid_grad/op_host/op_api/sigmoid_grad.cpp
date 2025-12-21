/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sigmoid_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SigmoidGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
inline static bool IsAiCoreSupport(const aclTensor *gradOutput)
{
    // sigmoidGrad只需要判断dtype
    return CheckType(gradOutput->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
inline static const aclTensor *SigmoidGradAiCore(const aclTensor *output, const aclTensor *gradOutput,
                                                 aclTensor *sigmoidGradInput, aclOpExecutor *executor)
{
    L0_DFX(SigmoidGradAiCore, output, gradOutput, sigmoidGradInput);
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCore SigmoidGrad算子加入任务队列
    // SigmoidGrad是算子的OpType，gradOutput、output是算子的输入，sigmoidGradInput是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(SigmoidGrad, OP_INPUT(output, gradOutput), OP_OUTPUT(sigmoidGradInput));
    return sigmoidGradInput;
}

// AICPU算子kernel
inline static const aclTensor *SigmoidGradAiCpu(const aclTensor *output, const aclTensor *gradOutput,
                                                aclTensor *sigmoidGradInput, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_AICPU_LAUNCHER_LIST，将AiCpu SigmoidGrad算子加入任务队列
    // SigmoidGrad是算子的OpType，gradOutput、output是算子的输入，sigmoidGradInput是算子的输出
    L0_DFX(SigmoidGradAiCpu, output, gradOutput, sigmoidGradInput);

    static internal::AicpuTaskSpace space("SigmoidGrad");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SigmoidGrad, OP_ATTR_NAMES({ "complex_conj" }), OP_INPUT(output, gradOutput),
                                          OP_OUTPUT(sigmoidGradInput), OP_ATTR(true));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return sigmoidGradInput;
}

const aclTensor *SigmoidGrad(const aclTensor *output, const aclTensor *gradOutput, aclOpExecutor *executor)
{
    auto sigmoidGradInput = executor->AllocTensor(gradOutput->GetViewShape(), gradOutput->GetDataType());
    if (IsAiCoreSupport(gradOutput)) {
        return SigmoidGradAiCore(output, gradOutput, sigmoidGradInput, executor);
    } else {
        return SigmoidGradAiCpu(output, gradOutput, sigmoidGradInput, executor);
    }
}
}
