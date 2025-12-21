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
 * \file aclnn_ctc_loss.cpp
 * \brief
 */
#include "aclnn_ctc_loss.h"
#include "ctc_loss_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t TARGET_MAX_DIM_NUM = 2;
constexpr size_t CTC_LOSS_V2_OUT_PARAM_NUM = 2;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t LOG_PROBS_DIM_T = 0;
constexpr size_t LOG_PROBS_3D_DIM_N = 1;
constexpr size_t LOG_PROBS_3D_DIM_C = 2;
constexpr size_t LOG_PROBS_2D_DIM_C = 1;
constexpr int64_t MAX_SYMBOL_SET_LEN_V3 = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 8180;
constexpr int64_t DOUBLE = 2;
constexpr int64_t CDIM = 2;
constexpr int64_t SDIM = 2;

// 根据API定义,需要列出log_prob&out参数所能支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LOGPROB_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_BF16_LOGPROB_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_DOUBLE};

// 根据API定义,需要列出target参数所能支持的dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_TARGET_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16};

static bool CheckNotNull(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetLengths, const aclTensor* negLogLikelihoodOut, const aclTensor* logAlphaOut)
{
    OP_CHECK_NULL(logProbs, return false);
    OP_CHECK_NULL(targets, return false);
    OP_CHECK_NULL(inputLengths, return false);
    OP_CHECK_NULL(targetLengths, return false);
    OP_CHECK_NULL(negLogLikelihoodOut, return false);
    OP_CHECK_NULL(logAlphaOut, return false);

    return true;
}

static bool CheckDtypeValid(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* negLogLikelihoodOut,
    const aclTensor* logAlphaOut)
{
    // 检查logProbs的数据类型是否在算子的支持列表内
    bool is910bSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    bool IsRegbaseSocVersion = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
    int64_t inputN = logProbs->GetViewShape().GetDim(1);
    int64_t inputC = logProbs->GetViewShape().GetDim(CDIM);
    int64_t inputS = (logAlphaOut->GetViewShape().GetDim(SDIM) - 1) / DOUBLE;
    bool shapeCheck = inputC > MAX_SYMBOL_SET_LEN_V3 || inputN > MAX_BATCH || inputS > MAX_LABEL_LEN;
    bool dtypeCheck =
        logProbs->GetDataType() == op::DataType::DT_BF16 || logProbs->GetDataType() == op::DataType::DT_FLOAT16;

    if (!is910bSocVersion && !IsRegbaseSocVersion) {
        OP_CHECK_DTYPE_NOT_SUPPORT(logProbs, DTYPE_SUPPORT_LOGPROB_LIST, return false);
    } else if (IsRegbaseSocVersion) {
        OP_CHECK_DTYPE_NOT_SUPPORT(logProbs, DTYPE_SUPPORT_BF16_LOGPROB_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(logProbs, DTYPE_SUPPORT_BF16_LOGPROB_LIST, return false);
        if (shapeCheck && dtypeCheck) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not supported now, we will support it later.");
            return false;
        }
    }

    // 检查targets的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(targets, DTYPE_SUPPORT_TARGET_LIST, return false);

    // 检查out的数据类型是否与logProbs相同
    OP_CHECK_DTYPE_NOT_MATCH(negLogLikelihoodOut, logProbs->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_MATCH(logAlphaOut, logProbs->GetDataType(), return false);

    return true;
}

static std::tuple<int64_t, int64_t> GetIntArrayMaxValueAndSum(const aclIntArray* intArrayValue)
{
    // 获取intArrayValue中的最大值
    int64_t maxLength = 0;

    // 获取intArrayValue的和
    int64_t sum = 0;
    for (size_t i = 0; i < intArrayValue->Size(); i++) {
        sum += static_cast<int64_t>((*intArrayValue)[i]);
        if (static_cast<int64_t>((*intArrayValue)[i]) > maxLength) {
            maxLength = static_cast<int64_t>((*intArrayValue)[i]);
        }
    }
    return std::tie(maxLength, sum);
}

static bool CheckTargetAndOutShapeParameterValue(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* targetLengths,
    const aclTensor* negLogLikelihoodOut, const aclTensor* logAlphaOut, int64_t& maxTargetLengths)
{
    auto targetLengthMaxAndSum = GetIntArrayMaxValueAndSum(targetLengths);
    maxTargetLengths = std::get<0>(targetLengthMaxAndSum);

    // 检查targets的shape，只能是1维或者2维
    if (targets->GetViewShape().GetDimNum() != 1 && targets->GetViewShape().GetDimNum() != TARGET_MAX_DIM_NUM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "DimNum of targets [%zu] must equal 1 or 2.", targets->GetViewShape().GetDimNum());
        return false;
    }

    // 当target的shape为2维,则必须满足shape(N,S),S >=maxTargetLengths
    if (targets->GetViewShape().GetDimNum() == TARGET_MAX_DIM_NUM) {
        int64_t inputN = 1;
        if (logProbs->GetViewShape().GetDimNum() == DIM_NUM_THREE) {
            inputN = logProbs->GetViewShape().GetDim(LOG_PROBS_3D_DIM_N);
        }
        if (targets->GetViewShape().GetDim(0) != inputN || targets->GetViewShape().GetDim(1) < maxTargetLengths) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Target's shape expect [%ld, >=%ld], but get current shape %s.", inputN,
                maxTargetLengths, op::ToString(targets->GetViewShape()).GetString());
            return false;
        }
    } else {
        // 当target的shape为1维,则必须满足shape(sum(targetLengths))
        int64_t targetLengthsSum = std::get<1>(targetLengthMaxAndSum);
        if (targets->GetViewShape().GetDim(0) != targetLengthsSum) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Target's shape expect [%ld], but get current shape %s.", targetLengthsSum,
                op::ToString(targets->GetViewShape()).GetString());
            return false;
        }
    }

    // 检查outNegLogLikelihood大小为N的一维Tensor
    auto outputShapeTuple = l0op::CtcLossNpuOutputShape(logProbs, maxTargetLengths);
    if (negLogLikelihoodOut->GetViewShape() != std::get<0>(outputShapeTuple)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of negLogLikelihoodOut %s must equal %s.",
            op::ToString(negLogLikelihoodOut->GetViewShape()).GetString(),
            op::ToString(std::get<0>(outputShapeTuple)).GetString());
        return false;
    }

    // 检查 logAlphaOut 大小为x,y,z的3维Tensor
    if (logAlphaOut->GetViewShape() != std::get<1>(outputShapeTuple)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of logAlphaOut %s must equal %s.",
            op::ToString(logAlphaOut->GetViewShape()).GetString(),
            op::ToString(std::get<1>(outputShapeTuple)).GetString());
        return false;
    }
    return true;
}

static bool CheckShapeAndParameterValue(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetLengths, int64_t blank, const aclTensor* negLogLikelihoodOut, const aclTensor* logAlphaOut,
    int64_t& maxTargetLengths)
{
    if (logProbs->GetViewShape().GetDim(0) == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "start (0) + length (1) exceeds dimension size (0).");
        return false;
    }

    // logProbs必须为3维或2维Tensor
    if (logProbs->GetViewShape().GetDimNum() > DIM_NUM_THREE || logProbs->GetViewShape().GetDimNum() < DIM_NUM_TWO) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "DimNum of logProbs [%zu] must equal 2 or 3.",
            logProbs->GetViewShape().GetDimNum());
        return false;
    }

    int64_t inputT = logProbs->GetViewShape().GetDim(LOG_PROBS_DIM_T);
    int64_t inputN = 1;
    int64_t inputC = logProbs->GetViewShape().GetDim(LOG_PROBS_2D_DIM_C);
    if (logProbs->GetViewShape().GetDimNum() == DIM_NUM_THREE) {
        inputN = logProbs->GetViewShape().GetDim(LOG_PROBS_3D_DIM_N);
        inputC = logProbs->GetViewShape().GetDim(LOG_PROBS_3D_DIM_C);
    }

    // 检查blank参数范围和reduction取值范围
    if (blank < 0 || blank >= inputC) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Attr parameter blank %ld must be greater than or equal to 0 and less than %ld.",
            blank, inputC);
        return false;
    }

    // 检查inputLengths数组的长度和数值的合法性
    const int64_t maxInputLengthValue = std::get<0>(GetIntArrayMaxValueAndSum(inputLengths));
    if (static_cast<int64_t>(inputLengths->Size()) != inputN || maxInputLengthValue > inputT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Inputlengths's size [%ld] must equal to [%ld], and max value [%ld] must be less than or "
            "equal to [%ld].",
            static_cast<int64_t>(inputLengths->Size()), inputN, maxInputLengthValue, inputT);
        return false;
    }

    // 检查targetLengths数组的长度合法性
    if (static_cast<int64_t>(targetLengths->Size()) != inputN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "targetLengths's size [%ld] must equal to [%ld].",
            static_cast<int64_t>(targetLengths->Size()), inputN);
        return false;
    }

    return CheckTargetAndOutShapeParameterValue(
        logProbs, targets, targetLengths, negLogLikelihoodOut, logAlphaOut, maxTargetLengths);
}

static aclnnStatus CheckParams(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetLengths, int64_t blank, const aclTensor* negLogLikelihoodOut, const aclTensor* logAlphaOut,
    int64_t& maxTargetLengths)
{
    // 1.检查输入shape是否满足要求
    CHECK_RET(
        CheckShapeAndParameterValue(
            logProbs, targets, inputLengths, targetLengths, blank, negLogLikelihoodOut, logAlphaOut, maxTargetLengths),
        ACLNN_ERR_PARAM_INVALID);

    // 2.检查输入的数据类型是否在API支持的数据类型范围之内,需要根据api定义校验
    CHECK_RET(CheckDtypeValid(logProbs, targets, negLogLikelihoodOut, logAlphaOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckTupleNullptr(std::tuple<aclTensor*, aclTensor*> tensorTuple)
{
    if (std::tuple_size<decltype(tensorTuple)>::value != CTC_LOSS_V2_OUT_PARAM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of tuple returned by CtcLossV2 is not 2.");
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

static const aclTensor* logProbsReshapeAndContiguous(const aclTensor* logProbs, aclOpExecutor* executor)
{
    auto logProbsContiguous = l0op::Contiguous(logProbs, executor);
    CHECK_RET(logProbsContiguous != nullptr, nullptr);
    if (logProbs->GetViewShape().GetDimNum() == DIM_NUM_THREE) {
        return logProbsContiguous;
    }

    int64_t inputT = logProbs->GetViewShape().GetDim(LOG_PROBS_DIM_T);
    int64_t inputN = 1;
    int64_t inputC = logProbs->GetViewShape().GetDim(LOG_PROBS_2D_DIM_C);

    int64_t logProbsShapeValue[3] = {inputT, inputN, inputC};
    aclIntArray* logProbsShape = executor->AllocIntArray(logProbsShapeValue, DIM_NUM_THREE);
    auto logProbsReshape = l0op::Reshape(logProbsContiguous, logProbsShape, executor);
    return logProbsReshape;
}

aclnnStatus aclnnCtcLossGetWorkspaceSize(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetLengths, int64_t blank, bool zeroInfinity, aclTensor* negLogLikelihoodOut,
    aclTensor* logAlphaOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnCtcLoss, DFX_IN(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        DFX_OUT(negLogLikelihoodOut, logAlphaOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1、检查三个入参参数是否为空指针
    CHECK_RET(
        CheckNotNull(logProbs, targets, inputLengths, targetLengths, negLogLikelihoodOut, logAlphaOut),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2、固定写法，参数检查
    int64_t maxTargetLengths = 0;
    auto ret = CheckParams(
        logProbs, targets, inputLengths, targetLengths, blank, negLogLikelihoodOut, logAlphaOut, maxTargetLengths);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果inputN为0，空Tensor，则直接返回空或者nan
    if (logProbs->GetViewShape().GetDim(1) == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入logProbs转换成连续的tensor
    auto logProbsContiguous = logProbsReshapeAndContiguous(logProbs, uniqueExecutor.get());
    CHECK_RET(logProbsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入targets转换成连续的tensor
    auto targetsContiguous = targets;
    if (!targets->IsEmpty()) {
        targetsContiguous = l0op::Contiguous(targets, uniqueExecutor.get());
        CHECK_RET(targetsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 如果targetsContiguous为Bool,float,float16类型，则转成int64
    auto targetsContiguousCasted = targetsContiguous;
    if (targetsContiguous->GetDataType() != op::DataType::DT_INT32 &&
        targetsContiguous->GetDataType() != op::DataType::DT_INT64) {
        targetsContiguousCasted = l0op::Cast(targetsContiguous, op::DataType::DT_INT64, uniqueExecutor.get());
        CHECK_RET(targetsContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将inputLengths和targetLengths转化成Tensor
    auto inputLengthsTensor =
        uniqueExecutor.get()->ConvertToTensor(inputLengths, targetsContiguousCasted->GetDataType());
    auto targetLengthsTensor =
        uniqueExecutor.get()->ConvertToTensor(targetLengths, targetsContiguousCasted->GetDataType());

    // 调用CTCLossV2算子kernel
    auto ctcLossV2OpOut = l0op::CtcLossV2(
        logProbsContiguous, targetsContiguousCasted, inputLengthsTensor, targetLengthsTensor, blank, zeroInfinity,
        maxTargetLengths, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(ctcLossV2OpOut), ACLNN_ERR_INNER_NULLPTR);
    auto negLogLikelihood = std::get<0>(ctcLossV2OpOut);
    auto logAlpha = std::get<1>(ctcLossV2OpOut);

    //  固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult1 = l0op::ViewCopy(negLogLikelihood, negLogLikelihoodOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult2 = l0op::ViewCopy(logAlpha, logAlphaOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //  固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCtcLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCtcLoss);

    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
