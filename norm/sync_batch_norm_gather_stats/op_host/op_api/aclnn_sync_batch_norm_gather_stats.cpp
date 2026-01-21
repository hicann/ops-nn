/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/level2_base.h"
#include "sync_batch_norm_gather_stats.h"
#include "opdev/platform.h"
#include "aclnn_sync_batch_norm_gather_stats.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t SYNC_BATCH_NORM_OUTPUT_NUM = 4;
constexpr size_t SYNC_BATCH_NORM_ZERO = 0;
constexpr size_t SYNC_BATCH_NORM_ONE = 1;
constexpr size_t SYNC_BATCH_NORM_TWO = 2;
constexpr size_t SYNC_BATCH_NORM_THREE = 3;
namespace {
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
} // namespace

static const std::initializer_list<op::DataType> ASCEND910_95_SAMPLT_COUNT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910_SAMPLT_COUNT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static bool CheckNotNull(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, aclTensor* batchMean, aclTensor* batchInvstd)
{
    OP_CHECK_NULL(totalSum, return false);
    OP_CHECK_NULL(totalSquareSum, return false);
    OP_CHECK_NULL(sampleCount, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(variance, return false);
    OP_CHECK_NULL(batchMean, return false);
    OP_CHECK_NULL(batchInvstd, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, const aclTensor* mean,
    const aclTensor* variance, const aclTensor* batchMean, const aclTensor* batchInvstd)
{
    auto dtypeSupportList = ASCEND910_DTYPE_SUPPORT_LIST;
    auto dtypeSampleCountSupportList = ASCEND910_SAMPLT_COUNT_DTYPE_SUPPORT_LIST;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        dtypeSupportList = ASCEND910_95_DTYPE_SUPPORT_LIST;
        dtypeSampleCountSupportList = ASCEND910_95_SAMPLT_COUNT_DTYPE_SUPPORT_LIST;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(totalSum, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(totalSquareSum, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sampleCount, dtypeSampleCountSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(variance, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batchMean, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batchInvstd, dtypeSupportList, return false);

    OP_CHECK_DTYPE_NOT_SAME(totalSum, totalSquareSum, return false);
    OP_CHECK_DTYPE_NOT_SAME(totalSum, mean, return false);
    OP_CHECK_DTYPE_NOT_SAME(totalSum, variance, return false);
    OP_CHECK_DTYPE_NOT_SAME(batchMean, batchInvstd, return false);
    return true;
}

static bool CheckFormat(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, const aclTensor* mean,
    const aclTensor* variance)
{
    if (op::IsPrivateFormat(totalSum->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSum format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(totalSquareSum->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSquareSum format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(sampleCount->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sampleCount format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(mean->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mean format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(variance->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "variance format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    return true;
}

static constexpr int TWO_DIMS = 2;
static constexpr int ONE_DIMS = 1;
static constexpr int DIM_NUM_0 = 0;
static constexpr int DIM_NUM_1 = 1;

static bool CheckShape(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, const aclTensor* mean,
    const aclTensor* variance, const aclTensor* batchMean, const aclTensor* batchInvstd)
{
    OP_CHECK_WRONG_DIMENSION(totalSum, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(totalSquareSum, TWO_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(sampleCount, ONE_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(mean, ONE_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(variance, ONE_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(batchMean, ONE_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(batchInvstd, ONE_DIMS, return false);

    op::Shape totalSumShape = totalSum->GetViewShape();
    op::Shape totalSquareSumShape = totalSquareSum->GetViewShape();
    op::Shape sampleCountShape = sampleCount->GetViewShape();
    op::Shape meanShape = mean->GetViewShape();
    op::Shape varianceShape = variance->GetViewShape();
    op::Shape batchMeanShape = batchMean->GetViewShape();
    op::Shape batchInvstdShape = batchInvstd->GetViewShape();

    if (totalSumShape.GetDim(DIM_NUM_0) != totalSquareSumShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim0 not same totalSquareSumShape dim0");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_1) != totalSquareSumShape.GetDim(DIM_NUM_1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim1 not same totalSquareSumShape dim1");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_0) != sampleCountShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim0 not same sampleCountShape dim0");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_1) != meanShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim1 not same meanShape dim0");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_1) != varianceShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim1 not same varianceShape dim0");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_1) != batchMeanShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim1 not same batchMeanShape dim0");
        return false;
    }

    if (totalSumShape.GetDim(DIM_NUM_1) != batchInvstdShape.GetDim(DIM_NUM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim1 not same batchInvstdShape dim0");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, aclTensor* batchMean, aclTensor* batchInvstd)
{
    CHECK_RET(
        CheckNotNull(totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, batchInvstd),
        ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(
        CheckDtypeValid(totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, batchInvstd),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(totalSum, totalSquareSum, sampleCount, mean, variance), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShape(totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, batchInvstd),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus SyncBatchNormGatherStatsFun(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, float momentum, float eps, aclTensor* batchMean, aclTensor* batchInvstd,
    aclOpExecutor* executor)
{
    // 固定写法，将输入mean转换成连续的tensor
    auto totalSumContiguous = l0op::Contiguous(totalSum, executor);
    CHECK_RET(totalSumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入variance转换成连续的tensor
    auto totalSquareSumContiguous = l0op::Contiguous(totalSquareSum, executor);
    CHECK_RET(totalSquareSumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mean转换成连续的tensor
    auto sampleCountContiguous = l0op::Contiguous(sampleCount, executor);
    CHECK_RET(sampleCountContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sampleCast = l0op::Cast(sampleCountContiguous, op::DataType::DT_INT32, executor);
    CHECK_RET(sampleCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入variance转换成连续的tensor
    auto meanContiguous = l0op::Contiguous(mean, executor);
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mean转换成连续的tensor
    auto varContiguous = l0op::Contiguous(variance, executor);
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanFinal = const_cast<aclTensor*>(meanContiguous);
    auto varFinal = const_cast<aclTensor*>(varContiguous);

    std::array<aclTensor*, SYNC_BATCH_NORM_OUTPUT_NUM> outTensor = l0op::SyncBatchNormGatherStats(
        totalSumContiguous, totalSquareSumContiguous, sampleCast, meanFinal, varFinal, momentum, eps, executor);
    CHECK_RET(outTensor[SYNC_BATCH_NORM_ZERO] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outTensor[SYNC_BATCH_NORM_ONE] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outTensor[SYNC_BATCH_NORM_TWO] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outTensor[SYNC_BATCH_NORM_THREE] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto batchMeanCast = l0op::Cast(outTensor[SYNC_BATCH_NORM_ZERO], batchMean->GetDataType(), executor);
    CHECK_RET(batchMeanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto batchMeanCopy = l0op::ViewCopy(batchMeanCast, batchMean, executor);
    CHECK_RET(batchMeanCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto batchInvstdCast = l0op::Cast(outTensor[SYNC_BATCH_NORM_ONE], batchInvstd->GetDataType(), executor);
    CHECK_RET(batchInvstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto batchInvstdCopy = l0op::ViewCopy(batchInvstdCast, batchInvstd, executor);
    CHECK_RET(batchInvstdCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanFinalCast = l0op::Cast(outTensor[SYNC_BATCH_NORM_TWO], mean->GetDataType(), executor);
    CHECK_RET(meanFinalCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanCopy = l0op::ViewCopy(meanFinalCast, mean, executor);
    CHECK_RET(meanCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto varFinalCast = l0op::Cast(outTensor[SYNC_BATCH_NORM_THREE], variance->GetDataType(), executor);
    CHECK_RET(varFinalCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto varCopy = l0op::ViewCopy(varFinalCast, variance, executor);
    CHECK_RET(varCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, float momentum, float eps, aclTensor* batchMean, aclTensor* batchInvstd,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnSyncBatchNormGatherStats, DFX_IN(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        DFX_OUT(batchMean, batchInvstd));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, batchInvstd);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 根据实际支持情况补充
    op::Shape totalSumShape = totalSum->GetViewShape();
    if (totalSumShape.GetDim(DIM_NUM_0) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "totalSumShape dim0 is zero");
        return ACLNN_ERR_PARAM_INVALID;
    } else if (totalSumShape.GetDim(DIM_NUM_1) == 0) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto result = SyncBatchNormGatherStatsFun(
        totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps, batchMean, batchInvstd,
        uniqueExecutor.get());
    CHECK_RET(result == ACLNN_SUCCESS, result);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSyncBatchNormGatherStats(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSyncBatchNormGatherStats);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif