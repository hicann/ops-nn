/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "logsoftmax_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LogSoftmaxV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *LogSoftmaxAiCore(const aclTensor *x, aclIntArray *dimList, aclTensor *logSoftmaxOut,
                                         aclOpExecutor *executor) {
  L0_DFX(LogSoftmaxAiCore, x, dimList, logSoftmaxOut);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogSoftmaxV2, OP_INPUT(x), OP_OUTPUT(logSoftmaxOut), OP_ATTR(dimList));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSoftmaxAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return logSoftmaxOut;
}

// AICPU算子kernel
static const aclTensor *LogSoftmaxAiCpu(const aclTensor *x, aclIntArray *dimList, aclTensor *logSoftmaxOut,
                                        aclOpExecutor *executor) {
  L0_DFX(LogSoftmaxAiCpu, x, dimList, logSoftmaxOut);

  static internal::AicpuTaskSpace space("LogSoftmaxV2");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LogSoftmaxV2, OP_ATTR_NAMES({"axes"}), OP_INPUT(x), OP_OUTPUT(logSoftmaxOut),
                                        OP_ATTR(dimList));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSoftmaxAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return logSoftmaxOut;
}

const aclTensor *LogSoftmaxV2(const aclTensor *x, int64_t dim, aclOpExecutor *executor) {
  FVector<int64_t> dimListVector{dim};
  auto dimList = executor->AllocIntArray(dimListVector.data(), 1);
  auto logSoftmaxOut = executor->AllocTensor(x->GetViewShape(), x->GetDataType());

  if (IsAiCoreSupport(x)) {
    return LogSoftmaxAiCore(x, dimList, logSoftmaxOut, executor);
  } else {
    return LogSoftmaxAiCpu(x, dimList, logSoftmaxOut, executor);
  }
}

}  // namespace l0op