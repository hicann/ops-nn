/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_cross_entropy_loss_with_max_sum.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedCrossEntropyLossWithMaxSum);

const std::tuple<aclTensor*,aclTensor*>FusedCrossEntropyLossWithMaxSum(const aclTensor* logitsMax, const aclTensor* sumExpLogits,
                                                        const aclTensor* predictedLogits, float labelSmoothing, const aclTensor* inputOptional,
                                                        const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional, aclTensor* lossOut,
                                                        aclTensor* softMaxOutOptional, aclOpExecutor *executor) {
  lossOut = executor->AllocTensor(logitsMax->GetViewShape(), logitsMax->GetDataType());
  if (vocabParallelLogitsOptional != nullptr) {
      softMaxOutOptional = executor->AllocTensor(vocabParallelLogitsOptional->GetViewShape(), logitsMax->GetDataType());
  } else {
    op::Shape softMaxOutShape;
    softMaxOutShape.AppendDim(0);
      softMaxOutOptional = executor->AllocTensor(
        softMaxOutShape, softMaxOutShape, logitsMax->GetDataType(), logitsMax->GetStorageFormat(),
        logitsMax->GetOriginalFormat());
  }

  L0_DFX(FusedCrossEntropyLossWithMaxSum, logitsMax, sumExpLogits, predictedLogits, labelSmoothing, 
            inputOptional, weightOptional, vocabParallelLogitsOptional, lossOut, softMaxOutOptional);

  ADD_TO_LAUNCHER_LIST_AICORE(FusedCrossEntropyLossWithMaxSum,
                              OP_INPUT(logitsMax, sumExpLogits, predictedLogits, inputOptional, weightOptional, vocabParallelLogitsOptional),
                              OP_OUTPUT(lossOut, softMaxOutOptional),
                              OP_ATTR(labelSmoothing));
  return std::tuple<aclTensor*, aclTensor*>(lossOut, softMaxOutOptional);
}
} // l0op
