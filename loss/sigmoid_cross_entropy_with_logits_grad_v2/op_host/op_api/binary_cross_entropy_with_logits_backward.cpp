/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary_cross_entropy_with_logits_backward.h"
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
OP_TYPE_REGISTER(SigmoidCrossEntropyWithLogitsGradV2);

const aclTensor* SigmoidCrossEntropyWithLogitsGradV2(const aclTensor* gradOutput, const aclTensor* self,
                                                     const aclTensor* target, const aclTensor* weightOptional,
                                                     const aclTensor* posWeightOptional, const std::string& reduction,
                                                     aclOpExecutor* executor) {
  L0_DFX(SigmoidCrossEntropyWithLogitsGradV2, gradOutput, self, target, weightOptional, posWeightOptional, reduction);

  auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
  auto ret = INFER_SHAPE(SigmoidCrossEntropyWithLogitsGradV2,
                         OP_INPUT(self, target, gradOutput, weightOptional, posWeightOptional), OP_OUTPUT(out),
                         OP_ATTR(reduction.c_str()));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
    return nullptr;
  }

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SigmoidCrossEntropyWithLogitsGradV2,
                                               OP_INPUT(self, target, gradOutput, weightOptional, posWeightOptional),
                                               OP_OUTPUT(out),
                                               OP_ATTR(reduction));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "SigmoidCrossEntropyWithLogitsGradV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}
}  // namespace l0op
