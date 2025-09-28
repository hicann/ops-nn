/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softshrink_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SoftShrinkGrad);

const aclTensor *SoftShrinkGrad(const aclTensor *gradOutput, const aclTensor *self,
                                float lambda, aclOpExecutor *executor) {
  auto gradInput = executor->AllocTensor(gradOutput->GetViewShape(), gradOutput->GetDataType(),
                                         gradOutput->GetViewFormat());
  CHECK_RET(gradInput != nullptr, nullptr);

  L0_DFX(SoftShrinkGrad, gradOutput, self, lambda, gradInput);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftShrinkGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(gradInput), OP_ATTR(lambda));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftShrinkGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return gradInput;
}
}  // namespace l0op