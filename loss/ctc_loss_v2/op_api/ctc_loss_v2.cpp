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
 * \file ctc_loss_v2.cpp
 * \brief
 */

#include "ctc_loss_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr size_t SYMBOL_SET_INDEX = 2;
constexpr size_t BATCH_INDEX = 1;
constexpr size_t TIME_INDEX = 0;
constexpr size_t LABEL_INDEX = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr int64_t MAX_SYMBOL_SET_LEN = 7500;
constexpr int64_t MAX_SYMBOL_SET_LEN_V3 = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 8180;
constexpr int64_t COEF = 2;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t FOUR_BYTE = 4;
constexpr int64_t EIGHT_BYTE = 8;
constexpr int64_t SMALL_UB_SIZE = 192 * 1024;
constexpr int64_t UB_SIZE = 256 * 1024;
constexpr int64_t RESERVED_UB_SIZE = 2 * 1024;
constexpr int64_t C_LOOP_BLOCK = 64;
constexpr int64_t C_LOOP_SUM_BLOCK = 8;
constexpr int64_t C_UB_NUM = 5;
constexpr int64_t BATCH_UB_NUM = 2;
constexpr int64_t TARGETS_UB_NUM = 24;
constexpr float ONE_FLOAT = 1.0f;
} // namespace

namespace l0op {
OP_TYPE_REGISTER(CTCLossV2);
OP_TYPE_REGISTER(CTCLossV3);

constexpr size_t LOG_PROBS_DIM_T = 0;
constexpr size_t LOG_PROBS_3D_DIM_N = 1;
static const std::string REDUCTION_MEAN = "mean";
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> V3_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FOUR_BYTE_DTYPE_LIST = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> EIGHT_BYTE_DTYPE_LIST = {op::DataType::DT_INT64};

static int64_t GetDtypeSize(const op::DataType dtype)
{
    if (op::CheckType(dtype, FOUR_BYTE_DTYPE_LIST)) {
        return FOUR_BYTE;
    }
    if (op::CheckType(dtype, EIGHT_BYTE_DTYPE_LIST)) {
        return EIGHT_BYTE;
    }
    return 0;
}

static bool IsregBaseAiCoreSupport(const aclTensor* logProbs)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND950) {
        return CheckType(logProbs->GetDataType(), V3_AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// 根据size判断算子是否支持走aicore CTCLossV2
static bool IsV2AiCoreSupport(
    const aclTensor* logProbs, const aclTensor* logAlpha, const aclTensor* targets, const aclTensor* inputLengthsTensor)
{
    auto logAlphaShape = logAlpha->GetViewShape();
    int64_t maxLabel = (logAlphaShape.GetDim(LABEL_INDEX) - 1) / COEF;
    if (maxLabel > MAX_LABEL_LEN) {
        return false;
    }
    auto logProbsShape = logProbs->GetViewShape();
    int64_t symbleSet = logProbsShape.GetDim(SYMBOL_SET_INDEX);
    int64_t batchSize = logProbsShape.GetDim(BATCH_INDEX);
    int64_t timeStep = logProbsShape.GetDim(TIME_INDEX);
    int64_t targetsDsize = GetDtypeSize(targets->GetDataType());
    int64_t inputLengthsDsize = GetDtypeSize(inputLengthsTensor->GetDataType());
    int64_t allDataSize = int64_t(
        symbleSet * (C_UB_NUM + ONE_FLOAT / C_LOOP_BLOCK + ONE_FLOAT / C_LOOP_SUM_BLOCK) * FP32_BYTES +
        timeStep * inputLengthsDsize + maxLabel * (TARGETS_UB_NUM * FP32_BYTES + targetsDsize));
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    int64_t ubSize = (socVersion >= SocVersion::ASCEND910B) ? SMALL_UB_SIZE : UB_SIZE;
    int64_t availableUbSize = ubSize - RESERVED_UB_SIZE;
    if (allDataSize >= availableUbSize) {
        return false;
    }
    if (batchSize * (targetsDsize + FP32_BYTES) >= availableUbSize) {
        return false;
    }
    return CheckType(logProbs->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// 根据size判断算子是否支持走aicore CTCLossV3
static bool IsV3AiCoreSupport(const aclTensor* logProbs, const aclTensor* logAlpha, const aclTensor* targets)
{
    auto logAlphaShape = logAlpha->GetViewShape();
    int64_t maxLabel = (logAlphaShape.GetDim(LABEL_INDEX) - 1) / COEF;
    if (maxLabel > MAX_LABEL_LEN) {
        return false;
    }
    auto logProbsShape = logProbs->GetViewShape();
    int64_t maxSymbol = logProbsShape.GetDim(SYMBOL_SET_INDEX);
    if (maxSymbol > MAX_SYMBOL_SET_LEN_V3) {
        return false;
    }
    int64_t batchSize = logProbsShape.GetDim(BATCH_INDEX);
    if (batchSize > MAX_BATCH) {
        return false;
    }
    auto targetsDtype = targets->GetDataType();
    if ((targetsDtype != op::DataType::DT_INT32) && (targetsDtype != op::DataType::DT_INT64)) {
        return false;
    }
    auto SocVersion = GetCurrentPlatformInfo().GetSocVersion();
    if ((SocVersion != SocVersion::ASCEND910B) && (SocVersion != SocVersion::ASCEND910_93)) {
        return false;
    }
    return CheckType(logProbs->GetDataType(), V3_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
const std::tuple<aclTensor*, aclTensor*> CtcLossAiCore(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, aclTensor* negLogLikelihood, aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor)
{
    L0_DFX(
        CtcLossAiCore, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank,
        zeroInfinity);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CTCLossV2, OP_INPUT(logProbs, targets, inputLengthsTensor, targetLengthsTensor),
        OP_OUTPUT(negLogLikelihood, logAlpha), OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CtcLossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(negLogLikelihood, logAlpha);
}

// CTCLossV3 AICORE算子kernel
const std::tuple<aclTensor*, aclTensor*> CtcLossV3AiCore(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, aclTensor* negLogLikelihood, aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor)
{
    L0_DFX(
        CtcLossV3AiCore, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank,
        zeroInfinity);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CTCLossV3, OP_INPUT(logProbs, targets, inputLengthsTensor, targetLengthsTensor),
        OP_OUTPUT(negLogLikelihood, logAlpha), OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CtcLossV3AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(negLogLikelihood, logAlpha);
}

// AICPU算子kernel
const std::tuple<aclTensor*, aclTensor*> CtcLossAiCpu(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, aclTensor* negLogLikelihood, aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor)
{
    L0_DFX(
        CtcLossAiCpu, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank,
        zeroInfinity);

    static internal::AicpuTaskSpace space("CTCLossV2");
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        CTCLossV2, OP_ATTR_NAMES({"blank", "reduction", "zero_infinity"}),
        OP_INPUT(logProbs, targets, inputLengthsTensor, targetLengthsTensor), OP_OUTPUT(negLogLikelihood, logAlpha),
        OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CtcLossAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tie(negLogLikelihood, logAlpha);
}

const std::tuple<op::Shape, op::Shape> CtcLossNpuOutputShape(const aclTensor* logProbs, int64_t maxLength)
{
    auto logProbsShape = logProbs->GetViewShape();
    int64_t inputT = logProbsShape.GetDim(LOG_PROBS_DIM_T);
    int64_t inputN = 1;
    if (logProbsShape.GetDimNum() == DIM_NUM_THREE) {
        inputN = logProbsShape.GetDim(LOG_PROBS_3D_DIM_N);
    }
    int64_t alphaTailSize = 2 * maxLength + 1;
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    int64_t alphaTailSizeAlign = (socVersion == SocVersion::ASCEND950) ? alphaTailSize : (alphaTailSize + 7) / 8 * 8;
    op::Shape logAlphaShape = {inputN, inputT, alphaTailSizeAlign};

    if (logProbsShape.GetDimNum() == DIM_NUM_THREE) {
        op::Shape negLogLikelihoodShape = {inputN};
        return std::tie(negLogLikelihoodShape, logAlphaShape);
    }

    op::Shape negLogLikelihoodShape = op::Shape({});
    return std::tie(negLogLikelihoodShape, logAlphaShape);
}

const std::tuple<aclTensor*, aclTensor*> CtcLossV2(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, int64_t blank, bool zeroInfinity, int64_t maxTargetLengths,
    aclOpExecutor* executor)
{
    // 计算输出Tensor的Shape
    auto outputShapeTuple = CtcLossNpuOutputShape(logProbs, maxTargetLengths);
    op::Shape negLogLikelihoodShape = std::get<0>(outputShapeTuple);
    op::Shape logAlphaShape = std::get<1>(outputShapeTuple);

    // 申请输出tensor的空间
    auto negLogLikelihood =
        executor->AllocTensor(negLogLikelihoodShape, logProbs->GetDataType(), op::Format::FORMAT_ND);
    auto logAlpha = executor->AllocTensor(logAlphaShape, logProbs->GetDataType(), op::Format::FORMAT_ND);
    if (IsregBaseAiCoreSupport(logProbs) && !targets->IsEmpty()) {
        return CtcLossAiCore(
            logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank, zeroInfinity,
            executor);
    } else if (IsV3AiCoreSupport(logProbs, logAlpha, targets) && !targets->IsEmpty()) {
        return CtcLossV3AiCore(
            logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank, zeroInfinity,
            executor);
    } else if (IsV2AiCoreSupport(logProbs, logAlpha, targets, inputLengthsTensor) && !targets->IsEmpty()) {
        return CtcLossAiCore(
            logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank, zeroInfinity,
            executor);
    } else {
        return CtcLossAiCpu(
            logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank, zeroInfinity,
            executor);
    }
}
} // namespace l0op
