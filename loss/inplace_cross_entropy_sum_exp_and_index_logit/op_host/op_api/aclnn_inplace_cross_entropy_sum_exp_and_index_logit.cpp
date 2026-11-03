/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "inplace_cross_entropy_sum_exp_and_index_logit.h"
#include "aclnn_inplace_cross_entropy_sum_exp_and_index_logit.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus InplaceCrossEntropySumExpAndIndexLogitCommonProcess(
    aclTensor* vocabParallelLogits, const aclTensor* target, const aclTensor* globalLogitsMax, int64_t vocabStartIndex,
    int64_t vocabEndIndex, const aclTensor* predictedLogits, const aclTensor* sumExpLogits,
    const aclTensor* targetOffset, const aclTensor* targetMask, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_COND(vocabParallelLogits != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vocabParallelLogits is nullptr.");
    CHECK_COND(target != nullptr, ACLNN_ERR_PARAM_NULLPTR, "target is nullptr.");
    CHECK_COND(globalLogitsMax != nullptr, ACLNN_ERR_PARAM_NULLPTR, "globalLogitsMax is nullptr.");
    CHECK_COND(predictedLogits != nullptr, ACLNN_ERR_PARAM_NULLPTR, "predictedLogits is nullptr.");
    CHECK_COND(sumExpLogits != nullptr, ACLNN_ERR_PARAM_NULLPTR, "sumExpLogits is nullptr.");
    CHECK_COND(targetOffset != nullptr, ACLNN_ERR_PARAM_NULLPTR, "targetOffset is nullptr.");
    CHECK_COND(targetMask != nullptr, ACLNN_ERR_PARAM_NULLPTR, "targetMask is nullptr.");
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // vocabParallelLogits：CreateView 零拷贝保留非连续 stride（不支持转连续，原地语义需要）
    aclTensor* logitsRef = uniqueExecutor->CreateView(
        vocabParallelLogits, vocabParallelLogits->GetViewShape(), vocabParallelLogits->GetStorageShape(),
        vocabParallelLogits->GetViewStrides(), vocabParallelLogits->GetViewOffset());
    CHECK_COND(logitsRef != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for vocabParallelLogits failed.");

    // target / globalLogitsMax 是只读小张量，直接转连续
    target = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_COND(target != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous target failed.");

    globalLogitsMax = l0op::Contiguous(globalLogitsMax, uniqueExecutor.get());
    CHECK_COND(globalLogitsMax != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous globalLogitsMax failed.");

    // 输出（predicted / sum_exp / offset / mask）由调用方分配，均为连续张量；
    // exp 不单独分配 —— 它被折叠成 logitsRef 本身（in-place 输出）
    bool ok = l0op::InplaceCrossEntropySumExpAndIndexLogit(logitsRef, target, globalLogitsMax, vocabStartIndex,
                                                           vocabEndIndex, predictedLogits, sumExpLogits, targetOffset,
                                                           targetMask, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnInplaceCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
    aclTensor* vocabParallelLogits, const aclTensor* target, const aclTensor* globalLogitsMax, int64_t vocabStartIndex,
    int64_t vocabEndIndex, const aclTensor* predictedLogits, const aclTensor* sumExpLogits,
    const aclTensor* targetOffset, const aclTensor* targetMask, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceCrossEntropySumExpAndIndexLogit,
                   DFX_IN(vocabParallelLogits, target, globalLogitsMax, vocabStartIndex, vocabEndIndex),
                   DFX_OUT(predictedLogits, sumExpLogits, targetOffset, targetMask));
    return InplaceCrossEntropySumExpAndIndexLogitCommonProcess(
        vocabParallelLogits, target, globalLogitsMax, vocabStartIndex, vocabEndIndex, predictedLogits, sumExpLogits,
        targetOffset, targetMask, workspaceSize, executor);
}

aclnnStatus aclnnInplaceCrossEntropySumExpAndIndexLogit(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceCrossEntropySumExpAndIndexLogit);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
