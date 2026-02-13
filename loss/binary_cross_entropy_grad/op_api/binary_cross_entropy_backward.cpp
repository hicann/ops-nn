/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary_cross_entropy_backward.h"
#include "opdev/format_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BinaryCrossEntropyGrad);

const aclTensor *BinaryCrossEntropyGrad(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional,
    const std::string &reduction, aclOpExecutor *executor) {
  L0_DFX(BinaryCrossEntropyGrad, gradOutput, self, target, weightOptional, reduction);

  auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  auto ret = INFER_SHAPE(BinaryCrossEntropyGrad, OP_INPUT(self, target, gradOutput, weightOptional), OP_OUTPUT(out),
                         OP_ATTR(reduction.c_str()));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
    return nullptr;
  }

  auto retAiCore = ADD_TO_LAUNCHER_LIST_AICORE(BinaryCrossEntropyGrad, OP_INPUT(self, target, gradOutput, weightOptional),
                                         OP_OUTPUT(out),
                                         OP_ATTR_NAMES({"reduction"}), OP_ATTR(reduction));
  OP_CHECK(retAiCore ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BinaryCrossEntropyGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),	 
     return nullptr);
  return out;
}
}
