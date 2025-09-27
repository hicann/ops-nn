/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multilabel_margin_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MultilabelMarginLoss);

const std::tuple<aclTensor*, aclTensor*> MultilabelMarginLoss(const aclTensor* self,
                                                              const aclTensor* target,
                                                              const std::string &reduction,
                                                              op::Shape shape, aclOpExecutor* executor) {
  L0_DFX(MultilabelMarginLoss, self, target, reduction);
  // 固定写法，创建OpExecutor
  auto out = executor->AllocTensor(shape, self->GetDataType(), self->GetStorageFormat());
  CHECK_RET(out != nullptr, (std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr)));
  auto isTarget = executor->AllocTensor(target->GetViewShape(), DataType::DT_INT32, target->GetStorageFormat());
  CHECK_RET(isTarget != nullptr, (std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr)));
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MultilabelMarginLoss, OP_INPUT(self, target),
                                         OP_ATTR(reduction), OP_OUTPUT(out, isTarget));
  if (ret !=  ACL_SUCCESS) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MultilabelMarginLossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
      return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return std::tuple<aclTensor*, aclTensor*>(out, isTarget);
}

}  // namespace l0op
