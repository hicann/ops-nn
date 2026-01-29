/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "batch_norm_backward_reduce.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/mul.h"
#include "level0/reduce_sum_op.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "aclnn_batch_norm_backward_reduce.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t DEFAULT_MASK_TAG_CNT = 3; // 3个bool类型标识对应输出变量是否需要输出
static constexpr size_t MASK_INDEX_FOR_SUM_DY_AND_XMU = 0;
static constexpr size_t MASK_INDEX_FOR_GRAD_WEIGHT = 1;
static constexpr size_t MASK_INDEX_FOR_GRAD_BIAS = 2;

#define DEFAULT_INPUT_TENSOR_PARAM_CNT 4
#define DEFAULT_OUTPUT_TENSOR_PARAM_CNT 4

typedef std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> TupleArrayOutCast;
typedef std::tuple<aclTensor*, aclTensor*> TupleArraySum;

typedef struct InputParamS {
    const aclTensor* param;
    const char* paramName;
} InputParamT;

typedef struct OutputParamS {
    const aclTensor* param;
    const char* paramName;
} OutputParamT;
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST{DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST{
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline size_t GetMaskIdx(size_t listIdx)
{
    if (listIdx < 2) { // 2: list index of output param sumDyXmu
        return MASK_INDEX_FOR_SUM_DY_AND_XMU;
    }

    if (listIdx == 2) { // 2: list index of output param sumDyXmu
        return MASK_INDEX_FOR_GRAD_WEIGHT;
    }

    return MASK_INDEX_FOR_GRAD_BIAS;
}

static inline bool CheckNotNull(const aclTensor* checkParam, const char* paramName)
{
    if (checkParam == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "expected a proper Tensor but got null for argument %s.", paramName);
        return false;
    }

    return true;
}

// 入参nullptr检查
static bool CheckInputNotNull(InputParamT* inputParamTensorList, size_t inputListSize)
{
    for (size_t i = 0; i < inputListSize; i++) {
        CHECK_RET(CheckNotNull(inputParamTensorList[i].param, inputParamTensorList[i].paramName), false);
    }

    return true;
}

// 出参nullptr检查，注意出参受mask控制，如果对应mask为false，则不输出出参，出参可以为空
namespace{
static bool CheckOutputNotNull(OutputParamT* outputParamTensorList, size_t outputListSize, const bool* outputMask)
{
    for (size_t i = 0; i < outputListSize; i++) {
        size_t maskIdx = GetMaskIdx(i);
        if (!outputMask[maskIdx]) {
            continue;
        }

        bool isNotNull = CheckNotNull(outputParamTensorList[i].param, outputParamTensorList[i].paramName);

        CHECK_RET(isNotNull, false);
    }

    return true;
}
}


static inline bool CheckDtypeInSupportList(const aclTensor* checkParam, const char* paramName, bool mask)
{
    auto dtypeList = GetDtypeSupportList();
    if (mask && !CheckType(checkParam->GetDataType(), dtypeList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s not implemented for %s, should be in dtype support list %s.", paramName,
            ToString(checkParam->GetDataType()).GetString(), ToString(dtypeList).GetString());
        return false;
    }

    return true;
}
static bool CheckDtypeValid(
    InputParamT* inputParamTensorList, size_t inputListSize, OutputParamT* outputParamTensorList, size_t outputListSize,
    const bool* outputMask)
{
    // 检查数据类型是否在支持列表内，out类型需要根据对应mask值来确定是否需要判断
    bool dtypeInList = true;
    for (size_t i = 0; i < inputListSize; i++) {
        dtypeInList = CheckDtypeInSupportList(inputParamTensorList[i].param, inputParamTensorList[i].paramName, true);
        CHECK_RET(dtypeInList, false);
    }

    // gradOut和input数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(
        inputParamTensorList[0].param, (inputParamTensorList[1].param)->GetDataType(), return false);
    for (size_t i = 0; i < outputListSize; i++) {
        size_t maskIdx = GetMaskIdx(i);
        bool mask = outputMask[maskIdx];
        dtypeInList = CheckDtypeInSupportList(outputParamTensorList[i].param, outputParamTensorList[i].paramName, mask);
        CHECK_RET(dtypeInList, false);
    }

    return true;
}

static inline bool CheckFormatEqual(
    const aclTensor* leftParam, const char* leftParamName, const aclTensor* rightParam, const char* rightParamName,
    bool mask)
{
    if (mask && leftParam->GetStorageFormat() != rightParam->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of %s and %s should be equal. self [%s], out [%s].", leftParamName,
            rightParamName, ToString(leftParam->GetStorageFormat()).GetString(),
            ToString(rightParam->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool CheckFormat(
    InputParamT* inputParamTensorList, size_t inputListSize, OutputParamT* outputParamTensorList, size_t outputListSize,
    const bool* outputMask)
{
    // 输入的格式需要一致
    if ((inputParamTensorList[0].param)->GetStorageFormat() != (inputParamTensorList[1].param)->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of gradOut and input should be equal. self [%s], out [%s].",
            ToString((inputParamTensorList[0].param)->GetStorageFormat()).GetString(),
            ToString((inputParamTensorList[1].param)->GetStorageFormat()).GetString());
        return false;
    }

    // 输入格式不能是私有格式
    for (size_t i = 0; i < inputListSize; i++) {
        if (IsPrivateFormat((inputParamTensorList[i].param)->GetStorageFormat())) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW;"
                "but %s format is %s, please check.",
                inputParamTensorList[i].paramName,
                ToString((inputParamTensorList[i].param)->GetStorageFormat()).GetString());
            return false;
        }
    }

    // 输出格式不能是私有格式
    for (size_t i = 0; i < outputListSize; i++) {
        size_t maskIdx = GetMaskIdx(i);
        bool mask = outputMask[maskIdx];
        if (mask && IsPrivateFormat((outputParamTensorList[i].param)->GetStorageFormat())) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW;"
                "but %s format is %s, please check.",
                outputParamTensorList[i].paramName,
                ToString((outputParamTensorList[i].param)->GetStorageFormat()).GetString());
            return false;
        }
    }

    return true;
}

static bool CheckShape(
    const aclTensor* gradOut, const aclTensor* input, OutputParamT* outputParamTensorList, size_t outputListSize,
    const bool* outputMask)
{
    // 所有输入的维度都不能低于2且不能超过8
    OP_CHECK_MIN_DIM(gradOut, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradOut, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MIN_DIM(input, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);

    auto dimC = input->GetViewShape()[1];
    // C维不能为空
    if (dimC == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "channel size of input should not be zero.");
        return false;
    }

    op::Shape expectShape = {dimC};
    for (size_t i = 0; i < outputListSize; i++) {
        size_t maskIdx = GetMaskIdx(i);
        bool mask = outputMask[maskIdx];
        if (mask && outputParamTensorList[i].param->GetViewShape() != expectShape) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "expected tensor for %s to have same size as %s, but got %s.",
                outputParamTensorList[i].paramName, op::ToString(expectShape).GetString(),
                op::ToString(outputParamTensorList[i].param->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const bool inputG, const bool weightG, const bool biasG, const aclTensor* sumDy, const aclTensor* sumDyXmu,
    const aclTensor* gradWeight, const aclTensor* gradBias)
{
    // 0. 参数组装
    InputParamT inputParamTensorList[DEFAULT_INPUT_TENSOR_PARAM_CNT] = {
        {gradOut, "gradOut"}, {input, "input"}, {mean, "mean"}, {invstd, "invstd"}};
    OutputParamT outputParamTensorList[DEFAULT_OUTPUT_TENSOR_PARAM_CNT] = {
        {sumDy, "sumDy"}, {sumDyXmu, "sumDyXmu"}, {gradWeight, "gradWeight"}, {gradBias, "gradBias"}};
    bool outputMask[DEFAULT_MASK_TAG_CNT] = {inputG, weightG, biasG};

    size_t inputListSize = sizeof(inputParamTensorList) / sizeof(struct InputParamS);
    size_t outputListSize = sizeof(outputParamTensorList) / sizeof(struct OutputParamS);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckInputNotNull(inputParamTensorList, inputListSize), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOutputNotNull(outputParamTensorList, outputListSize, outputMask), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    bool ret = CheckDtypeValid(inputParamTensorList, inputListSize, outputParamTensorList, outputListSize, outputMask);
    CHECK_RET(ret, ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    ret = CheckFormat(inputParamTensorList, inputListSize, outputParamTensorList, outputListSize, outputMask);
    CHECK_RET(ret, ACLNN_ERR_PARAM_INVALID);

    // 4. 检查Shape是否支持
    ret = CheckShape(gradOut, input, outputParamTensorList, outputListSize, outputMask);
    CHECK_RET(ret, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus InputParamContiguousAndCast(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, TupleArrayOutCast& outCast, aclOpExecutor* executor)
{
    // 输入如果非连续，需要转换
    auto gradOutContiguous = l0op::Contiguous(gradOut, executor);
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanContiguous = l0op::Contiguous(mean, executor);
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invstdContiguous = l0op::Contiguous(invstd, executor);
    CHECK_RET(invstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 类型转换，都转换成Float类型进行计算
    auto gradOutCast = l0op::Cast(gradOutContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputCast = l0op::Cast(inputContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(inputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanCast = l0op::Cast(meanContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invstdCast = l0op::Cast(invstdContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(invstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightCast = l0op::Cast(weightContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 存转换后的值
    std::get<0>(outCast) = const_cast<aclTensor*>(gradOutCast);
    std::get<1>(outCast) = const_cast<aclTensor*>(inputCast);  // 1: input cast
    std::get<2>(outCast) = const_cast<aclTensor*>(meanCast);   // 2: mean cast
    std::get<3>(outCast) = const_cast<aclTensor*>(invstdCast); // 3: invstd cast
    std::get<4>(outCast) = const_cast<aclTensor*>(weightCast); // 4: weight cast

    return ACLNN_SUCCESS;
}

static inline aclnnStatus SingleOutParamCastAndViewCopy(
    const aclTensor* outCast, aclTensor* out, aclOpExecutor* executor)
{
    // reverse cast
    auto outRevCast = l0op::Cast(outCast, out->GetDataType(), executor);
    CHECK_RET(outRevCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // viewcopy to out tensor
    auto viewCopyResult = l0op::ViewCopy(outRevCast, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus OutputParamCastAndViewCopy(
    const aclTensor* sumDyOut, const aclTensor* sumDyXmuOut, const aclTensor* gradWeightOut, const bool inputG,
    const bool weightG, const bool biasG, aclTensor* sumDy, aclTensor* sumDyXmu, aclTensor* gradWeight,
    aclTensor* gradBias, aclOpExecutor* executor)
{
    aclnnStatus outCopyRet = ACLNN_SUCCESS;
    if (inputG) {
        outCopyRet = SingleOutParamCastAndViewCopy(sumDyOut, sumDy, executor);
        CHECK_RET(outCopyRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        outCopyRet = SingleOutParamCastAndViewCopy(sumDyXmuOut, sumDyXmu, executor);
        CHECK_RET(outCopyRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    if (weightG) {
        outCopyRet = SingleOutParamCastAndViewCopy(gradWeightOut, gradWeight, executor);
        CHECK_RET(outCopyRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    if (biasG) {
        outCopyRet = SingleOutParamCastAndViewCopy(sumDyOut, gradBias, executor);
        CHECK_RET(outCopyRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static bool CheckSyncResultNotNull(TupleArraySum outArray)
{
    if (std::tuple_size<decltype(outArray)>::value != 2) { // 2: for sumDyXmu and gradWeight output
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "sync return %lu out tensors, less than 2",
            std::tuple_size<decltype(outArray)>::value);
        return false;
    }

    return (std::get<0>(outArray) != nullptr && std::get<1>(outArray) != nullptr);
}

static aclnnStatus aclnnBatchNormReduceBackwardForSumDy(
    const aclTensor* gradOut, const aclTensor* input, TupleArraySum& sumOut, aclOpExecutor* executor)
{
    // 计算(dl/dy)*x
    auto mulRet = l0op::Mul(gradOut, input, executor);
    CHECK_RET(mulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 计算sum((dl/dy)*x)
    size_t inputDim = input->GetViewShape().GetDimNum();
    int64_t appendDim[inputDim - 1];
    uint64_t dimIdx = 0;
    for (uint64_t i = 0; i < inputDim; i++) {
        if (i == 1) { // 1: C dim
            continue;
        }
        appendDim[dimIdx++] = i;
    }
    aclIntArray* dim = executor->AllocIntArray(appendDim, inputDim - 1);
    CHECK_RET(dim != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto reduceSumDyXRet = l0op::ReduceSumOp(mulRet, dim, false, executor);
    CHECK_RET(reduceSumDyXRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 计算sum(dl/dy)
    auto reduceSumDyRet = l0op::ReduceSumOp(gradOut, dim, false, executor);
    CHECK_RET(reduceSumDyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 存储sumDy和sumDyX
    std::get<0>(sumOut) = const_cast<aclTensor*>(reduceSumDyRet);
    std::get<1>(sumOut) = const_cast<aclTensor*>(reduceSumDyXRet);

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnBatchNormReduceBackwardNpuImpl(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const bool inputG, const bool weightG, const bool biasG, aclTensor* sumDy,
    aclTensor* sumDyXmu, aclTensor* gradWeight, aclTensor* gradBias, aclOpExecutor* executor)
{
    // 非连续转换和类型转换
    TupleArrayOutCast outCast;
    auto ret = InputParamContiguousAndCast(gradOut, input, mean, invstd, weight, outCast, executor);
    CHECK_RET((ret == ACLNN_SUCCESS), ret);

    aclTensor* gradOutCast = std::get<0>(outCast);
    aclTensor* inputCast = std::get<1>(outCast);  // 1:input
    aclTensor* meanCast = std::get<2>(outCast);   // 2: mean
    aclTensor* invstdCast = std::get<3>(outCast); // 3: invstd+

    // 调用ReduceSum和Mul计算sumDy和sumDyX
    TupleArraySum sumOut;
    ret = aclnnBatchNormReduceBackwardForSumDy(gradOutCast, inputCast, sumOut, executor);
    CHECK_RET((ret == ACLNN_SUCCESS), ret);

    aclTensor* sumDyTmp = std::get<0>(sumOut);
    aclTensor* sumDyX = std::get<1>(sumOut);
    // 调用l0算子BatchNormReduceBackward进行计算
    auto outArray = l0op::SyncBatchNormBackwardReduce(sumDyTmp, sumDyX, meanCast, invstdCast, executor);
    CHECK_RET(CheckSyncResultNotNull(outArray), ACLNN_ERR_INNER_NULLPTR);

    // 整理输出结果
    aclTensor* sumDyXmuOut = std::get<0>(outArray);
    CHECK_RET(sumDyXmuOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor* gradWeightOut = std::get<1>(outArray);
    CHECK_RET(gradWeightOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ret = OutputParamCastAndViewCopy(
        sumDyTmp, sumDyXmuOut, gradWeightOut, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias, executor);
    CHECK_RET((ret == ACLNN_SUCCESS), ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus OutputParamFillZeroForEmptyTensor(
    const aclTensor* input, bool inputG, bool weightG, bool biasG, aclTensor* sumDy,
    aclTensor* sumDyXmu, aclTensor* gradWeight, aclTensor* gradBias, aclOpExecutor* executor)
{
    auto dimC = input->GetViewShape()[1];
    aclnnStatus outFillRet = ACLNN_SUCCESS;

    // C维不能为空
    if (dimC != 0 && GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        // 空Tensor情况下，输出都为0
        aclTensor* fillZeroTensorTmp = FillScalar(dimC, 0, executor);
        CHECK_RET(fillZeroTensorTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        outFillRet = OutputParamCastAndViewCopy(
            fillZeroTensorTmp, fillZeroTensorTmp, fillZeroTensorTmp, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias,
            executor);
        CHECK_RET((outFillRet == ACLNN_SUCCESS), outFillRet);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const bool inputG, const bool weightG, const bool biasG, aclTensor* sumDy,
    aclTensor* sumDyXmu, aclTensor* gradWeight, aclTensor* gradBias, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnBatchNormReduceBackward, DFX_IN(gradOut, input, mean, invstd, weight, inputG, weightG, biasG),
        DFX_OUT(sumDy, sumDyXmu, gradWeight, gradBias));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, input, mean, invstd, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (gradOut->IsEmpty() || input->IsEmpty() || mean->IsEmpty() || invstd->IsEmpty()) {
        auto fillZeroRet = OutputParamFillZeroForEmptyTensor(
            input, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias, uniqueExecutor.get());
        CHECK_RET(fillZeroRet == ACLNN_SUCCESS, fillZeroRet);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 可选入参weight处理
    if (weight == nullptr) {
        weight = FillScalar(input->GetViewShape()[1], 1, uniqueExecutor.get());
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 开始BN reduce反向
    auto bnBwReduceRet = aclnnBatchNormReduceBackwardNpuImpl(
        gradOut, input, mean, invstd, weight, inputG, weightG, biasG, sumDy, sumDyXmu, gradWeight, gradBias,
        uniqueExecutor.get());
    CHECK_RET(bnBwReduceRet == ACLNN_SUCCESS, bnBwReduceRet);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormReduceBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormReduceBackward);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
