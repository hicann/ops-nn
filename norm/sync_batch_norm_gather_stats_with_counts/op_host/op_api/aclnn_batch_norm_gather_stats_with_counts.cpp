/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../norm_common/op_host/norm_tensor_util.h"
#include "aclnn_kernels/cast.h"
#include "level0/fill.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/broadcast_to.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "level0/reduce_sum_op.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/reduce_mean_with_count.h"
#include "level0/sync_bn_training_update.h"
#include "sync_batch_norm_gather_stats_with_counts.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_batch_norm_gather_stats_with_counts.h"

namespace op {
aclTensor* FillVector(const aclIntArray* shapes, int value, aclOpExecutor* executor)
{
    const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT32);
    const aclScalar* valueScalar = executor->AllocScalar(value);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, op::DataType::DT_FLOAT);
    auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
    if (fillTensor == nullptr) {
        return nullptr;
    }
    fillTensor = l0op::ReFormat(fillTensor, op::Format::FORMAT_ND);
    return const_cast<aclTensor*>(fillTensor);
}

aclTensor* BroadCastTensor(const op::Shape dstShape, const aclTensor* src, aclOpExecutor* executor)
{
    auto dstTensor = executor->AllocTensor(dstShape, src->GetDataType());
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(dstShape);
    auto shape =
        executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(), static_cast<op::DataType>(ACL_INT64));
    auto result = l0op::BroadcastTo(src, dstTensor, shape, executor);
    return const_cast<aclTensor*>(result);
}
} // namespace op

using namespace op;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const int64_t MEAN_INDEX = 0;
static const int64_t INVSTD_INDEX = 1;
static const int64_t RUNNING_VAR_INDEX = 2;
static const int64_t OUTPUT_TENSOR_NUM = 3;

static inline bool CheckSocVersionGe910B(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}
static bool CheckNotNull(
    const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, const aclTensor* counts,
    const aclTensor* meanAll, const aclTensor* invstdAll)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(invstd, return false);
    OP_CHECK_NULL(counts, return false);
    OP_CHECK_NULL(meanAll, return false);
    OP_CHECK_NULL(invstdAll, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* counts, const aclTensor* meanAll, const aclTensor* invstdAll)
{
    bool is910BSocVersion = CheckSocVersionGe910B();
    const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_910B_LIST : DTYPE_SUPPORT_910_LIST;

    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(invstd, DTYPE_SUPPORT_LIST, return false);
    if (runningMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningMean, DTYPE_SUPPORT_LIST, return false);
    }
    if (runningVar != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningVar, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(counts, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(meanAll, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(invstdAll, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, const aclTensor* counts)
{
    if (op::IsPrivateFormat(input->GetStorageFormat()) || op::IsPrivateFormat(mean->GetStorageFormat()) ||
        op::IsPrivateFormat(invstd->GetStorageFormat()) || op::IsPrivateFormat(counts->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND,NCHW,NCL,NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* counts, const aclTensor* meanAll, const aclTensor* invstdAll)
{
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(input, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_WRONG_DIMENSION(mean, BN_MIN_SUPPORT_DIMS_NUMS, return false)
    OP_CHECK_WRONG_DIMENSION(invstd, BN_MIN_SUPPORT_DIMS_NUMS, return false)
    if (mean->GetViewShape()[0] != invstd->GetViewShape()[0] || counts->GetViewShape()[0] != mean->GetViewShape()[0]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "first dimension of mean, invstd and count should be the same, but got "
            "mean: %s, invstd: %s, counts: %s.",
            op::ToString(mean->GetViewShape()).GetString(), op::ToString(invstd->GetViewShape()).GetString(),
            op::ToString(counts->GetViewShape()).GetString());
        return false;
    }
    int64_t dimC = input->GetViewShape()[1];
    if (mean->GetViewShape()[1] != dimC || invstd->GetViewShape()[1] != dimC) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "second dimension of mean/invstd should be the channel size of input %ld, but got"
            "mean: %s, invstd: %s.",
            dimC, op::ToString(mean->GetViewShape()).GetString(), op::ToString(invstd->GetViewShape()).GetString());
        return false;
    }
    op::Shape expectShape = {dimC};
    if (runningMean != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(runningMean, expectShape, return false);
    }
    if (runningVar != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(runningVar, expectShape, return false);
    }

    if (counts->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim of counts should be one.");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanAll, expectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(invstdAll, expectShape, return false);
    return true;
}

static aclnnStatus CheckBatchNormGatherStatsWithCountsParams(
    const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, aclTensor* runningMean,
    aclTensor* runningVar, const aclTensor* counts, aclTensor* meanAll, aclTensor* invstdAll)
{
    CHECK_COND(
        CheckNotNull(input, mean, invstd, counts, meanAll, invstdAll), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");
    CHECK_COND(
        CheckDtypeValid(input, mean, invstd, runningMean, runningVar, counts, meanAll, invstdAll),
        ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
    CHECK_COND(CheckFormat(input, mean, invstd, counts), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");
    CHECK_COND(
        CheckShape(input, mean, invstd, runningMean, runningVar, counts, meanAll, invstdAll), ACLNN_ERR_PARAM_INVALID,
        "CheckShape failed!");
    return ACLNN_SUCCESS;
}

static const aclTensor* CalculateRunningTensor(const aclTensor* runningTensor, int64_t dimC, aclOpExecutor* executor)
{
    const int64_t shapes[] = {1, dimC};
    aclIntArray* shapeArray = executor->AllocIntArray(shapes, sizeof(shapes) / sizeof(int64_t));
    if (runningTensor == nullptr) {
        auto result = FillVector(shapeArray, 0, executor);
        CHECK_RET(result != nullptr, nullptr);
        return result;
    }
    const int64_t appendDims[1] = {0};
    aclIntArray* appendDimsArray = executor->AllocIntArray(appendDims, 1);
    auto runningTensorContiguous = l0op::Contiguous(runningTensor, executor);
    CHECK_RET(runningTensorContiguous != nullptr, nullptr);
    auto runningTensorCast = l0op::Cast(runningTensorContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(runningTensorCast != nullptr, nullptr);
    auto runningTensorMid = l0op::UnsqueezeNd(runningTensorCast, appendDimsArray, executor);
    CHECK_RET(runningTensorMid != nullptr, nullptr);
    auto result = l0op::ReFormat(runningTensorMid, Format::FORMAT_ND);
    CHECK_RET(result != nullptr, nullptr);
    return result;
}

static std::array<const aclTensor*, OUTPUT_TENSOR_NUM> GatherStatsWithCounts(
    const aclTensor* mean, const aclTensor* invstd, const aclTensor* counts, const aclTensor* runningVarMid,
    double momentum, double eps, aclOpExecutor* executor)
{
    std::array<const aclTensor*, OUTPUT_TENSOR_NUM> nulls = {nullptr, nullptr, nullptr};
    auto meanContiguous = l0op::Contiguous(mean, executor);
    CHECK_RET(meanContiguous != nullptr, nulls);

    auto meanCast = l0op::Cast(meanContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(meanCast != nullptr, nulls);

    auto invstdContiguous = l0op::Contiguous(invstd, executor);
    CHECK_RET(invstdContiguous != nullptr, nulls);

    auto invstdCast = l0op::Cast(invstdContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(invstdCast != nullptr, nulls);

    auto countContiguous = l0op::Contiguous(counts, executor);
    CHECK_RET(countContiguous != nullptr, nulls);

    auto countCast = l0op::Cast(countContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(countCast != nullptr, nulls);

    const int64_t countShape[1] = {-1};
    aclIntArray* countShapeArray = executor->AllocIntArray(countShape, 1);
    auto countUnsqueeze = l0op::UnsqueezeNd(countCast, countShapeArray, executor);
    CHECK_RET(countUnsqueeze != nullptr, nulls);

    auto countBroadcast = BroadCastTensor(invstdCast->GetViewShape(), countUnsqueeze, executor);
    CHECK_RET(countBroadcast != nullptr, nulls);

    const int64_t reduceAxes[1] = {0};
    aclIntArray* reduceAxesArray = executor->AllocIntArray(reduceAxes, 1);
    auto countAllSum = l0op::ReduceSumOp(countBroadcast, reduceAxesArray, true, executor);
    CHECK_RET(countAllSum != nullptr, nulls);

    auto countAllSumBroadcast = BroadCastTensor(countBroadcast->GetViewShape(), countAllSum, executor);
    CHECK_RET(countAllSumBroadcast != nullptr, nulls);

    auto meanAllOutput =
        l0op::ReduceMeanWithCount(meanCast, countBroadcast, countAllSumBroadcast, reduceAxesArray, true, executor);
    CHECK_RET(meanAllOutput != nullptr, nulls);

    auto meanAllBroadcast = BroadCastTensor(meanCast->GetViewShape(), meanAllOutput, executor);
    CHECK_RET(meanAllBroadcast != nullptr, nulls);

    auto batchNormOut = l0op::SyncBatchNormGatherStatsWithCounts(
        meanCast, invstdCast, countBroadcast, meanAllBroadcast, countAllSum, runningVarMid,
        static_cast<float>(momentum), static_cast<float>(eps), executor);
    return {meanAllOutput, batchNormOut[0], batchNormOut[1]};
}

static aclnnStatus CopyRunningTensorToDst(
    aclTensor* runningMeanDst, aclTensor* runningVarDst, const aclTensor* meanAllOutput,
    const aclTensor* runningMeanMid, const aclTensor* runningVarOutput, double momentum, aclOpExecutor* executor)
{
    if (runningMeanDst == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t appendDims[1] = {0};
    aclIntArray* appendDimsArray = executor->AllocIntArray(appendDims, 1);

    aclTensor* runningMean_ = const_cast<aclTensor*>(runningMeanMid);
    const aclTensor* runningMeanOutput =
        l0op::SyncBNTrainingUpdate(meanAllOutput, runningMean_, static_cast<float>(momentum), executor);
    CHECK_RET(runningMeanOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto runningMeanCast_ = runningMeanOutput;
    auto runningVarCast_ = runningVarOutput;
    if (runningMeanOutput->GetDataType() != runningMeanDst->GetDataType()) {
        runningMeanCast_ = l0op::Cast(runningMeanOutput, runningMeanDst->GetDataType(), executor);
        CHECK_RET(runningMeanCast_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (runningVarOutput->GetDataType() != runningVarDst->GetDataType()) {
        runningVarCast_ = l0op::Cast(runningVarOutput, runningVarDst->GetDataType(), executor);
        CHECK_RET(runningVarCast_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto runningVarSqueeze = l0op::SqueezeNd(runningVarCast_, appendDimsArray, executor);
    CHECK_RET(runningVarSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarViewCopyResult = l0op::ViewCopy(runningVarSqueeze, runningVarDst, executor);
    CHECK_RET(runningVarViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanSqueeze = l0op::SqueezeNd(runningMeanCast_, appendDimsArray, executor);
    CHECK_RET(runningMeanSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanViewCopyResult = l0op::ViewCopy(runningMeanSqueeze, runningMeanDst, executor);
    CHECK_RET(runningMeanViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyMeanAndInvstdToDst(
    const aclTensor* meanAllOutput, const aclTensor* invstdAllOutput, aclTensor* meanAll, aclTensor* invstdAll,
    aclOpExecutor* executor)
{
    auto meanAllCast = meanAllOutput;
    auto invstdAllCast = invstdAllOutput;

    meanAllCast = l0op::Cast(meanAllOutput, meanAll->GetDataType(), executor);
    CHECK_RET(meanAllCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    invstdAllCast = l0op::Cast(invstdAllOutput, invstdAll->GetDataType(), executor);
    CHECK_RET(invstdAllCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t appendDims[1] = {0};
    aclIntArray* appendDimsArray = executor->AllocIntArray(appendDims, 1);

    auto meanAllSqueeze = l0op::SqueezeNd(meanAllCast, appendDimsArray, executor);
    CHECK_RET(meanAllSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanAllViewCopyResult = l0op::ViewCopy(meanAllSqueeze, meanAll, executor);
    CHECK_RET(meanAllViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstdAllSqueeze = l0op::SqueezeNd(invstdAllCast, appendDimsArray, executor);
    CHECK_RET(invstdAllSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstdAllViewCopyResult = l0op::ViewCopy(invstdAllSqueeze, invstdAll, executor);
    CHECK_RET(invstdAllViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(
    const aclTensor* input, const aclTensor* mean, const aclTensor* invstd, aclTensor* runningMean,
    aclTensor* runningVar, double momentum, double eps, const aclTensor* counts, aclTensor* meanAll,
    aclTensor* invstdAll, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnBatchNormGatherStatsWithCounts,
        DFX_IN(input, mean, invstd, runningMean, runningVar, momentum, eps, counts), DFX_OUT(meanAll, invstdAll));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckBatchNormGatherStatsWithCountsParams(
        input, mean, invstd, runningMean, runningVar, counts, meanAll, invstdAll);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (mean->IsEmpty()) {
        ret = op::ProcessEmptyTensorWithValue(meanAll, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = op::ProcessEmptyTensorWithValue(invstdAll, std::numeric_limits<float>::quiet_NaN(), uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool isFullyFp16 = false;
    if (input->GetDataType() == mean->GetDataType() && input->GetDataType() == op::DataType::DT_FLOAT16) {
        isFullyFp16 = true;
    }

    const aclTensor* runningMeanMid =
        CalculateRunningTensor(runningMean, input->GetViewShape()[1], uniqueExecutor.get());
    CHECK_RET(runningMeanMid != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* runningVarMid = CalculateRunningTensor(runningVar, input->GetViewShape()[1], uniqueExecutor.get());
    CHECK_RET(runningVarMid != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gatherResult = GatherStatsWithCounts(mean, invstd, counts, runningVarMid, momentum, eps, uniqueExecutor.get());
    auto meanAllOutput = gatherResult[MEAN_INDEX];
    CHECK_RET(meanAllOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invstdOutput = gatherResult[INVSTD_INDEX];
    CHECK_RET(invstdOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto runningVarOutput = gatherResult[RUNNING_VAR_INDEX];
    CHECK_RET(runningVarOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = CopyRunningTensorToDst(
        runningMean, runningVar, meanAllOutput, runningMeanMid, runningVarOutput, momentum, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CopyMeanAndInvstdToDst(meanAllOutput, invstdOutput, meanAll, invstdAll, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormGatherStatsWithCounts(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormGatherStatsWithCounts);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
