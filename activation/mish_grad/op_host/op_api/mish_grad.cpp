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
 * \file mish_grad.cpp
 * \brief
 */

#include "mish_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MishGrad);

const aclTensor* MishGrad(const aclTensor* gradOutput, const aclTensor* self, aclOpExecutor* executor) {
  L0_DFX(MishGrad, gradOutput, self);
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetStorageFormat());
  CHECK_RET(out != nullptr, nullptr);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MishGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MishGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op
