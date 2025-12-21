/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_linear_online_max_sum.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedLinearOnlineMaxSum);

static const int64_t BITS_PER_B8 = 8;

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> FusedLinearOnlineMaxSum(
    const aclTensor* input, const aclTensor* weight, const aclTensor* target, int64_t vocabStartIndex,
    int64_t vocabEndIndex, bool vocabParallelLogitsOutFlag, aclOpExecutor* executor) {
    auto logitsMaxLocal = executor->AllocTensor(
        target->GetStorageShape(), target->GetOriginalShape(), DataType::DT_FLOAT, target->GetStorageFormat(),
        target->GetOriginalFormat());
    
    auto sumExpLogitsLocal = executor->AllocTensor(
        target->GetStorageShape(), target->GetOriginalShape(), DataType::DT_FLOAT, target->GetStorageFormat(),
        target->GetOriginalFormat());
    
    auto predictedLogitsLocal = executor->AllocTensor(
        target->GetStorageShape(), target->GetOriginalShape(), DataType::DT_FLOAT, target->GetStorageFormat(),
        target->GetOriginalFormat());

    op::Shape targetMaskShape;
    int64_t dimSize0 = target->GetViewShape().GetDim(0);
    targetMaskShape.AppendDim((dimSize0 + BITS_PER_B8 - 1) / BITS_PER_B8);
    auto targetMask = executor->AllocTensor(
        targetMaskShape, targetMaskShape, DataType::DT_UINT8, target->GetStorageFormat(), target->GetOriginalFormat());

    auto maskedTarget = executor->AllocTensor(
        target->GetStorageShape(), target->GetOriginalShape(), target->GetDataType(), target->GetStorageFormat(),
        target->GetOriginalFormat());

    op::Shape vocabParallelLogitsOutShape;
    if (vocabParallelLogitsOutFlag) {
        int64_t dimSize1 = weight->GetViewShape().GetDim(0);
        vocabParallelLogitsOutShape.AppendDim(dimSize0);
        vocabParallelLogitsOutShape.AppendDim(dimSize1);
    } else {
        vocabParallelLogitsOutShape.AppendDim(0);
    }
    auto vocabParallelLogitsOut = executor->AllocTensor(
        vocabParallelLogitsOutShape, vocabParallelLogitsOutShape, input->GetDataType(), input->GetStorageFormat(),
        input->GetOriginalFormat());

    L0_DFX(FusedLinearOnlineMaxSum, input, weight, target, vocabStartIndex, vocabEndIndex, vocabParallelLogitsOutFlag,
           logitsMaxLocal, sumExpLogitsLocal, predictedLogitsLocal, targetMask, maskedTarget, vocabParallelLogitsOut);

    ADD_TO_LAUNCHER_LIST_AICORE(FusedLinearOnlineMaxSum, OP_INPUT(input, weight, target),
        OP_OUTPUT(logitsMaxLocal, sumExpLogitsLocal, predictedLogitsLocal, targetMask, maskedTarget,
                  vocabParallelLogitsOut),
        OP_ATTR(vocabStartIndex, vocabEndIndex, vocabParallelLogitsOutFlag));

    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
        logitsMaxLocal, sumExpLogitsLocal, predictedLogitsLocal, targetMask, maskedTarget, vocabParallelLogitsOut);
}
}  // namespace l0op
