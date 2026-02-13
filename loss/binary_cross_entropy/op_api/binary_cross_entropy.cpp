/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary_cross_entropy.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BinaryCrossEntropy);

const aclTensor *BinaryCrossEntropy(const aclTensor *self,
                      const aclTensor *target,
                      const aclTensor *weight,
                      const std::string &reduction,
                      aclOpExecutor *executor) {
  L0_DFX(BinaryCrossEntropy, self, target, weight, reduction);
  auto result = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  INFER_SHAPE(BinaryCrossEntropy, OP_INPUT(self, target, weight), OP_OUTPUT(result), OP_ATTR(reduction));

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(BinaryCrossEntropy,
                                               OP_INPUT(self, target, weight),
                                               OP_OUTPUT(result),
                                               OP_ATTR(reduction));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "BinaryCrossEntropy ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return result;
}

}