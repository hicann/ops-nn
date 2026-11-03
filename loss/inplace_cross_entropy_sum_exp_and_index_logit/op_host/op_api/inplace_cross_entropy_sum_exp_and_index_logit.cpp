/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_cross_entropy_sum_exp_and_index_logit.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(InplaceCrossEntropySumExpAndIndexLogit);

bool InplaceCrossEntropySumExpAndIndexLogit(aclTensor* vocabParallelLogits, const aclTensor* target,
                                            const aclTensor* globalLogitsMax, int64_t vocabStartIndex,
                                            int64_t vocabEndIndex, const aclTensor* predictedLogits,
                                            const aclTensor* sumExpLogits, const aclTensor* targetOffset,
                                            const aclTensor* targetMask, aclOpExecutor* executor)
{
    L0_DFX(InplaceCrossEntropySumExpAndIndexLogit, vocabParallelLogits, target, globalLogitsMax, vocabStartIndex,
           vocabEndIndex, predictedLogits, sumExpLogits, targetOffset, targetMask);
    // GE 原型保持 5 个 OUTPUT（含 index 2 的 vocab_parallel_logits in-place 别名）
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        InplaceCrossEntropySumExpAndIndexLogit, OP_INPUT(vocabParallelLogits, target, globalLogitsMax),
        OP_OUTPUT(predictedLogits, sumExpLogits, vocabParallelLogits, targetOffset, targetMask),
        OP_ATTR(vocabStartIndex, vocabEndIndex));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return false,
                                         "InplaceCrossEntropySumExpAndIndexLogit AddToLauncherList failed.");

    return true;
}

} // namespace l0op
