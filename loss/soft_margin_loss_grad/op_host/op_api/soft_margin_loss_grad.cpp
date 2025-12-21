/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "soft_margin_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SoftMarginLossGrad);

const aclTensor *SoftMarginLossGrad(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor) {
  L0_DFX(SoftMarginLossGrad, gradOutput, self, target, reduction)
  op::Shape broadcastShape1;
  op::Shape broadcastShape;
  if (!BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOutput tensor shape:%s and self tensor shape:%s can't broadcast.",
            ToString(gradOutput->GetViewShape()).GetString(), ToString(self->GetViewShape()).GetString());
    return nullptr;
  }
  if (!BroadcastInferShape(target->GetViewShape(), broadcastShape1, broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "GradOutput tensor shape:%s self tensor shape:%s target tensor shape:%s can't broadcast.",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString(),
            op::ToString(target->GetViewShape()).GetString());
    return nullptr;
  }
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftMarginLossGrad, OP_INPUT(self, target, gradOutput), OP_OUTPUT(out),
                                         OP_ATTR(reduction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftMarginLossGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op