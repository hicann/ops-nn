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
 * \file kl_div_loss_grad.cpp
 * \brief
 */

#include "kl_div_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(KlDivLossGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};
// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  // KlDivLossGrad只需要判断dtype
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *KlDivLossGradAiCore(const aclTensor *gradOutput, const aclTensor *self,
                                            const aclTensor *target, const char* reduction,
                                            bool logTarget, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(KlDivLossGradAiCore, gradOutput, self, target, reduction, logTarget);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(KlDivLossGrad, OP_INPUT(gradOutput, self, target),
                                         OP_OUTPUT(out), OP_ATTR(reduction, logTarget));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KlDivLossGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);

  return out;
}

const aclTensor *KlDivLossGrad(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target,
                               const char* reduction, bool logTarget, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetStorageFormat());
  if (IsAiCoreSupport(self)) {
    // 只走aicore
    return KlDivLossGradAiCore(gradOutput, self, target, reduction, logTarget, out, executor);
  }
  return out;
}
}  // namespace l0op
