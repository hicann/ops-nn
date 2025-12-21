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
 * \file mse_loss_grad.cpp
 * \brief
 */

#include "mse_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MseLossGrad);
OP_TYPE_REGISTER(MseLossGradV2);

const aclTensor *MseLossGradV1(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out) {
  OP_LOGD("Entering L0 MseLossGradV1");
  L0_DFX(MseLossGradV1, gradOutput, self, target, reduction)
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MseLossGrad, OP_INPUT(self, target, gradOutput), OP_OUTPUT(out), OP_ATTR(reduction));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MseLossGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *MseLossGradV2(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out) {
  OP_LOGD("Entering L0 MseLossGradV2");
  L0_DFX(MseLossGradV2, gradOutput, self, target, reduction)
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MseLossGradV2, OP_INPUT(self, target, gradOutput), OP_OUTPUT(out), OP_ATTR(reduction));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MseLossGradV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *MseLossGrad(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor) {
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
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
  if ((socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
       socVersion == SocVersion::ASCEND310P) && // MseLossGradV2算子仅支持910B/910_93和310P芯片
       self->GetViewShape() == target->GetViewShape() && gradOutput->GetViewShape() == target->GetViewShape() &&
       self->GetViewFormat() == target->GetViewFormat() && gradOutput->GetViewFormat() == target->GetViewFormat()) {
    return MseLossGradV2(gradOutput, self, target, reduction, executor, out);
  }
  return MseLossGradV1(gradOutput, self, target, reduction, executor, out);
}
}  // namespace l0op