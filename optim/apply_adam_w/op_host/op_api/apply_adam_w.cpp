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
 * \file apply_adam_w.cpp
 * \brief
 */
#include "apply_adam_w.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ApplyAdamW);

// AICORE算子kernel
std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> ApplyAdamW(const aclTensor* varRef,
                                                                            const aclTensor* mRef,
                                                                            const aclTensor* vRef,
                                                                            const aclTensor* beta1Power,
                                                                            const aclTensor* beta2Power,
                                                                            const aclTensor* lr,
                                                                            const aclTensor* weightDecay,
                                                                            const aclTensor* beta1,
                                                                            const aclTensor* beta2,
                                                                            const aclTensor* eps,
                                                                            const aclTensor* grad,
                                                                            const aclTensor* maxGradNormOptional,
                                                                            bool amsgrad, bool maximize,
                                                                            aclOpExecutor *executor) {
  L0_DFX(ApplyAdamW, varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
         maxGradNormOptional, amsgrad, maximize);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdamW,
                                               OP_INPUT(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay,
                                                        beta1, beta2, eps, grad, maxGradNormOptional),
                                               OP_OUTPUT(varRef, mRef, vRef),
                                               OP_ATTR(amsgrad, maximize));
  if (retAicore != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdamW ADD_TO_LAUNCHER_LIST_AICORE failed.");
  }
  return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(varRef, mRef, vRef);
}

}  // namespace l0op
