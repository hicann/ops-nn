/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "softplus.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SoftplusV2);

const aclTensor *SoftplusV2(const aclTensor *self, float beta, float threshold,
                            aclOpExecutor *executor) {
  auto softplusOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  CHECK_RET(softplusOut != nullptr, nullptr);

  // AICORE算子kernel
  L0_DFX(SoftplusV2, self, beta, threshold, softplusOut);

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore SoftplusV2算子加入任务队列
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SoftplusV2,
                                               OP_INPUT(self),
                                               OP_OUTPUT(softplusOut),
                                               OP_ATTR(beta, threshold));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "SoftplusV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return softplusOut;
}
}  // namespace l0op