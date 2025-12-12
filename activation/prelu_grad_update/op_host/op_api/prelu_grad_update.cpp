/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "prelu_grad_update.h"
#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"


using namespace op;

namespace l0op {
OP_TYPE_REGISTER(PReluGradUpdate);

//无Aicpu分支
const std::tuple<aclTensor*, aclTensor*> PReluGradUpdate(const aclTensor *gradOutput,
                                                          const aclTensor *self,
                                                          const aclTensor *weight,
                                                          aclOpExecutor *executor) {
  // AICORE算子kernel
  L0_DFX(PReluGradUpdate, gradOutput, self, weight);

  auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  auto update = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore PReluGradUpdate算子加入任务队列
  ADD_TO_LAUNCHER_LIST_AICORE(PReluGradUpdate,
                              OP_INPUT(gradOutput, self, weight),
                              OP_OUTPUT(gradInput, update));
  return std::tie(gradInput, update);
}
}  // namespace l0op
