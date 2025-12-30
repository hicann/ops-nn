/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swish.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Swish);

static const aclTensor *SiluAiCore(const aclTensor *self, aclTensor *out, float scale, aclOpExecutor *executor) {
  L0_DFX(SiluAiCore, self, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Swish, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(scale));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SwishAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

const aclTensor *Swish(const aclTensor *self, float scale, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  return SiluAiCore(self, out, scale, executor);
}
}  // namespace l0op
