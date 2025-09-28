/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "softplus_v2_grad.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SoftplusV2Grad);

const aclTensor *SoftplusV2Grad(const aclTensor *grad_output, const aclTensor *self,
                                float beta, float threshold, aclOpExecutor *executor) {
  // AICORE算子kernel
  L0_DFX(SoftplusV2Grad, grad_output, self, beta, threshold);

  auto grad_input = executor->AllocTensor(grad_output->GetViewShape(), grad_output->GetDataType());
  CHECK_RET(grad_input != nullptr, nullptr);

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore SoftplusV2Grad算子加入任务队列
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftplusV2Grad,
                                         OP_INPUT(grad_output, self),
                                         OP_OUTPUT(grad_input),
                                         OP_ATTR(beta, threshold));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftplusV2GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return grad_input;
}
}  // namespace l0op
