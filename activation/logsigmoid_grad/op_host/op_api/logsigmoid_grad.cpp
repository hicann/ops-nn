/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logsigmoid_grad.cpp
 * \brief
 */
#include "logsigmoid_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LogSigmoidGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *LogSigmoidGradAiCore(const aclTensor *gradOuput, const aclTensor *self, aclTensor *gradInput,
                                             aclOpExecutor *executor) {
    L0_DFX(LogSigmoidGradAiCore, gradOuput, self, gradInput);

    ADD_TO_LAUNCHER_LIST_AICORE(LogSigmoidGrad, OP_INPUT(gradOuput, self), OP_OUTPUT(gradInput));
    return gradInput;
}

const aclTensor *LogSigmoidGrad(const aclTensor *gradOutput, const aclTensor *self, aclOpExecutor *executor) {
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

    if (IsAiCoreSupport(self)) {
        return LogSigmoidGradAiCore(gradOutput, self, gradInput, executor);
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSigmoidGrad do not have AiCpu kernel.");
        return nullptr;
    }
}
}  // namespace l0op
