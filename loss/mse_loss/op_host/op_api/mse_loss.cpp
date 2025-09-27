/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss.cpp
 * \brief
 */
#include "mse_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MseLoss);
OP_TYPE_REGISTER(MSELossV2);

static const std::string REDUCTION_NONE = "none";

const aclTensor* MSELossV2(const aclTensor *self, const aclTensor *target, const std::string &reduction,
                         aclOpExecutor *executor) {
  L0_DFX(MSELossV2, self, target)
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
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MSELossV2, OP_INPUT(self, target), OP_OUTPUT(lossOut), OP_ATTR(reduction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MSELossV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return lossOut;
}

const aclTensor* MseLossV1(const aclTensor *self, const aclTensor *target, const std::string &reduction,
                         aclOpExecutor *executor) {
  L0_DFX(MseLossV1, self, target)
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
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MseLoss, OP_INPUT(self, target), OP_OUTPUT(lossOut), OP_ATTR(reduction));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MseLossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return lossOut;
}

const aclTensor *MseLoss(const aclTensor *self, const aclTensor *target, const std::string &reduction,
                         aclOpExecutor *executor) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if ((socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
       socVersion == SocVersion::ASCEND310P) && // MSELossV2算子仅支持910B/910_93和310P芯片
      self->GetViewShape() == target->GetViewShape() && self->GetViewFormat() == target->GetViewFormat() &&
      (self->GetViewFormat() == ge::FORMAT_ND || self->GetViewFormat() == ge::FORMAT_NCL)) {
    OP_LOGD("MSELossV2");
    return MSELossV2(self, target, reduction, executor);
  }
  OP_LOGD("MSELossV1");
  return MseLossV1(self, target, reduction, executor);
}
}  // namespace l0op
