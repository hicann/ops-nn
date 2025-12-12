/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "selugrad.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SeluGrad);

static const aclTensor *SeluGradAiCore(const aclTensor *gradOutput, const aclTensor *reslut, aclTensor *gradInput, aclOpExecutor *executor) {
  L0_DFX(SeluGradAiCore, gradOutput, reslut, gradInput);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SeluGrad, OP_INPUT(gradOutput, reslut), OP_OUTPUT(gradInput));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "SeluGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return gradInput;
}

const aclTensor *SeluGrad(const aclTensor *gradOutput, const aclTensor *reslut, aclOpExecutor *executor) {
  auto gradInput = executor->AllocTensor(gradOutput->GetViewShape(), gradOutput->GetDataType());
  return SeluGradAiCore(gradOutput, reslut, gradInput, executor);
}
}  // namespace l0op
