/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "soft_margin_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"


using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SoftMarginLoss);

static const std::string REDUCTION_NONE = "none";

const aclTensor *SoftMarginLoss(const aclTensor *self, const aclTensor *target, const std::string &reduction,
                                aclOpExecutor *executor) {
  L0_DFX(SoftMarginLoss, self, target)
  op::Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self tensor shape:%s and target tensor shape:%s can't broadcast.",
            ToString(self->GetViewShape()).GetString(), ToString(target->GetViewShape()).GetString());
    return nullptr;
  }
  aclTensor *lossOut = nullptr;
  if (reduction == REDUCTION_NONE) {
    lossOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  } else {
    lossOut = executor->AllocTensor({}, self->GetDataType());
  }
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftMarginLoss, OP_INPUT(self, target), OP_OUTPUT(lossOut), OP_ATTR(reduction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftMarginLossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return lossOut;
}
}  // namespace l0op