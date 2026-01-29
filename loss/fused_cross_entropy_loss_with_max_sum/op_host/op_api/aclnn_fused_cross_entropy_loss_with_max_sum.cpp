/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_fused_cross_entropy_loss_with_max_sum.h"
#include "fused_cross_entropy_loss_with_max_sum.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static const int64_t DIM0 = 0;
static const int64_t LOGISTS_MAX = 1;
static const int64_t VOCAB_DIM = 2;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_LOGITS = {DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_VOCAB = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits,
    const aclTensor* lossOut)
{
    OP_CHECK_NULL(logitsMax, return false);
    OP_CHECK_NULL(sumExpLogits, return false);
    OP_CHECK_NULL(predictedLogits, return false);
    OP_CHECK_NULL(lossOut, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits,
    const aclTensor* vocabParallelLogitsOptional, const aclTensor* lossOut, const aclTensor* softMaxOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(logitsMax, DTYPE_SUPPORT_LIST_LOGITS, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sumExpLogits, DTYPE_SUPPORT_LIST_LOGITS, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(predictedLogits, DTYPE_SUPPORT_LIST_LOGITS, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(lossOut, DTYPE_SUPPORT_LIST_LOGITS, return false);

    if (vocabParallelLogitsOptional != nullptr) {
        // 检查vocabParallelLogitsOptional的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(vocabParallelLogitsOptional, DTYPE_SUPPORT_LIST_VOCAB, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(softMaxOutOptional, DTYPE_SUPPORT_LIST_LOGITS, return false);
    }

    return true;
}

static inline bool CheckShape(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits,
    const aclTensor* vocabParallelLogitsOptional, const aclTensor* lossOut, const aclTensor* softMaxOutOptional)
{
    OP_CHECK_MAX_DIM(logitsMax, LOGISTS_MAX, return false);
    OP_CHECK_MIN_DIM(logitsMax, LOGISTS_MAX, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(logitsMax, sumExpLogits, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(logitsMax, predictedLogits, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(logitsMax, lossOut, return false);

    if (vocabParallelLogitsOptional != nullptr) {
        OP_CHECK_MAX_DIM(vocabParallelLogitsOptional, VOCAB_DIM, return false);
        OP_CHECK_MIN_DIM(vocabParallelLogitsOptional, VOCAB_DIM, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(vocabParallelLogitsOptional, softMaxOutOptional, return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits, float labelSmoothing,
    [[maybe_unused]] const aclTensor* inputOptional, [[maybe_unused]] const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional,
    aclTensor* lossOut, aclTensor* softMaxOutOptional)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(logitsMax, sumExpLogits, predictedLogits, lossOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(
            logitsMax, sumExpLogits, predictedLogits, vocabParallelLogitsOptional, lossOut, softMaxOutOptional),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(
        CheckShape(logitsMax, sumExpLogits, predictedLogits, vocabParallelLogitsOptional, lossOut, softMaxOutOptional),
        ACLNN_ERR_PARAM_INVALID);

    if (labelSmoothing != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Currently, only labelSmoothing=0 are supported.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits, float labelSmoothing,
    const aclTensor* inputOptional, const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional,
    aclTensor* lossOut, aclTensor* softMaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnFusedCrossEntropyLossWithMaxSum,
        DFX_IN(
            logitsMax, sumExpLogits, predictedLogits, labelSmoothing, inputOptional, weightOptional,
            vocabParallelLogitsOptional),
        DFX_OUT(lossOut, softMaxOutOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(
        logitsMax, sumExpLogits, predictedLogits, labelSmoothing, inputOptional, weightOptional,
        vocabParallelLogitsOptional, lossOut, softMaxOutOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    //  logitsMax如果非连续，需要转连续
    auto logitsMaxContiguous = l0op::Contiguous(logitsMax, uniqueExecutor.get());
    CHECK_RET(logitsMaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  sumExpLogits如果非连续，需要转连续
    auto sumExpLogitsContiguous = l0op::Contiguous(sumExpLogits, uniqueExecutor.get());
    CHECK_RET(sumExpLogitsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  predictedLogits如果非连续，需要转连续
    auto predictedLogitsContiguous = l0op::Contiguous(predictedLogits, uniqueExecutor.get());
    CHECK_RET(predictedLogitsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* vocabParallelLogitsOptionalContiguous = nullptr;
    if (vocabParallelLogitsOptional != nullptr) {
        //  vocabParallelLogitsOptional如果非连续，需要转连续
        vocabParallelLogitsOptionalContiguous = l0op::Contiguous(vocabParallelLogitsOptional, uniqueExecutor.get());
        CHECK_RET(vocabParallelLogitsOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子FusedCrossEntropyLossWithMaxSum进行计算
    auto result = l0op::FusedCrossEntropyLossWithMaxSum(
        logitsMaxContiguous, sumExpLogitsContiguous, predictedLogitsContiguous, labelSmoothing, inputOptional,
        weightOptional, vocabParallelLogitsOptionalContiguous, lossOut, softMaxOutOptional, uniqueExecutor.get());
    auto viewCopylossOutResult = l0op::ViewCopy(std::get<0>(result), lossOut, uniqueExecutor.get());
    CHECK_RET(viewCopylossOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (vocabParallelLogitsOptional != nullptr) {
        auto viewCopysoftMaxOutOptionalResult =
            l0op::ViewCopy(std::get<1>(result), softMaxOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopysoftMaxOutOptionalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedCrossEntropyLossWithMaxSum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
