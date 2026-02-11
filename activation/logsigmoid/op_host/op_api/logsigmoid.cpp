/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logsigmoid.cpp
 * \brief
 */
#include "logsigmoid.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LogSigmoid);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *LogSigmoidAiCore(const aclTensor *x, aclTensor *LogSigmoidOut, aclOpExecutor *executor) {
    L0_DFX(LogSigmoidAiCore, x, LogSigmoidOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogSigmoid, OP_INPUT(x), OP_OUTPUT(LogSigmoidOut));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSigmoidAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return LogSigmoidOut;
}

const aclTensor *LogSigmoid(const aclTensor *x, aclOpExecutor *executor) {
    auto LogSigmoidOut = executor->AllocTensor(x->GetViewShape(), x->GetDataType());

    if (IsAiCoreSupport(x)) {
        return LogSigmoidAiCore(x, LogSigmoidOut, executor);
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSigmoid do not have AiCpu kernel.");
        return nullptr;
    }
}
}  // namespace l0op
