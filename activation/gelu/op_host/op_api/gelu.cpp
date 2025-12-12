/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gelu.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(Gelu);

const aclTensor *Gelu(const aclTensor *self, aclOpExecutor *executor) {
  L0_DFX(Gelu, self);

  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Gelu, OP_INPUT(self), OP_OUTPUT(out));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                      "Gelu ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}
} // namespace l0op