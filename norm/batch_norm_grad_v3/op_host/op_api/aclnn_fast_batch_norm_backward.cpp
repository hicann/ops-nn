/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../norm_common/op_host/norm_tensor_util.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/unsqueeze.h"
#include "level0/mul.h"
#include "level0/reduce_sum_op.h"
#include "level0/add.h"
#include "level0/sub.h"
#include "level0/pow.h"
#include "level0/div.h"
#include "level0/unsqueeze.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "batch_norm_backward.h"
#include "aclnn_fast_batch_norm_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t GRAD_WEIGHT_INDEX = 1;
constexpr size_t GRAD_BIAS_INDEX = 2;
constexpr size_t MIN_BN_DIMS = 2;
constexpr size_t MAX_BN_DIMS = 5;
constexpr size_t BN2D_INPUT_DIMS = 4;
constexpr size_t UPDATE_GRAD_RESULT_CNT = 2;
constexpr size_t BN_GRAD_V3_RESULT_CNT = 3;
constexpr size_t BN_TRANSPOSE_N_LIMIT = 1000;
constexpr size_t BN_TRANSPOSE_HW_LIMIT = 32;
constexpr size_t ONE = 1;
constexpr size_t TWO = 2;
namespace {
static constexpr float EXP_INV_SQRT = -0.5f;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool isBatchNormSupportNcdhw(void)
{
    return (
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93));
}

static bool CheckInputNotNull(const aclTensor* gradOut, const aclTensor* input)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(input, return false);

    return true;
}

static inline bool isBatchNormSupportAscendC(void)
{
    bool socVersionSupport =
        ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
         (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93));

    return socVersionSupport;
}

static bool CheckMaskNotNull(
    const aclTensor* gradInput, const aclTensor* gradWeight, const aclTensor* gradBias, const aclBoolArray* outputMask)
{
    OP_CHECK_NULL(outputMask, return false);
    if (outputMask->Size() < GRAD_WEIGHT_INDEX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputMask size should not less than 1.");
        return false;
    }
    if ((*outputMask)[0]) {
        OP_CHECK_NULL(gradInput, return false);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK_NULL(gradWeight, return false);
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK_NULL(gradBias, return false);
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* gradOut, const aclTensor* input)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);
    OP_CHECK_DTYPE_NOT_SAME(input, gradOut, return false);

    return true;
}

static bool CheckGradDtypeValid(
    const aclTensor* gradInput, const aclTensor* gradWeight, const aclTensor* gradBias, const aclBoolArray* outputMask)
{
    const auto& supportList = GetDtypeSupportList();
    if ((*outputMask)[0]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, supportList, return false);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradWeight, supportList, return false);
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradBias, supportList, return false);
    }

    return true;
}

static bool CheckOtherDtypeValid(
    const aclTensor* weight, const aclTensor* runningMean, const aclTensor* runningVar, const aclTensor* saveMean,
    const aclTensor* saveInvstd)
{
    const auto& supportList = GetDtypeSupportList();
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weight, supportList, return false);
    }
    if (runningMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningMean, supportList, return false);
    }
    if (runningVar != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningVar, supportList, return false);
    }
    if (saveMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(saveMean, supportList, return false);
    }
    if (saveInvstd != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(saveInvstd, supportList, return false);
    }
    return true;
}

static bool CheckFormat(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput, const aclBoolArray* outputMask)
{
    if (gradOut->GetStorageFormat() != input->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of gradOut and input should be equal, gradOut [%s], input [%s].",
            op::ToString(gradOut->GetStorageFormat()).GetString(), op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    if ((*outputMask)[0] && gradOut->GetStorageFormat() != gradInput->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of gradOut and gradInput should be equal, gradOut [%s], gradInput [%s].",
            op::ToString(gradOut->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOut->GetStorageFormat()) || op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NC, NCL, NCHW, NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput, const aclBoolArray* outputMask)
{
    const int max_check_nums = 8;
    OP_CHECK_MAX_DIM(input, max_check_nums, return false);
    OP_CHECK_MIN_DIM(input, MIN_BN_DIMS, return false);
    OP_CHECK_MAX_DIM(gradOut, max_check_nums, return false);
    OP_CHECK_MIN_DIM(gradOut, MIN_BN_DIMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOut, input, return false);

    if ((*outputMask)[0]) {
        OP_CHECK_MAX_DIM(gradInput, max_check_nums, return false);
        OP_CHECK_MIN_DIM(gradInput, MIN_BN_DIMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(gradInput, gradOut, return false);
    }
    return true;
}

static bool CheckOtherShape(
    int dimC, const aclTensor* weight, const aclTensor* runningMean, const aclTensor* runningVar)
{
    if (weight != nullptr && (weight->GetViewShape().GetDimNum() != 1 || weight->GetViewShape()[0] != dimC)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dim of weight should be one and shape is channel num of input[%d], but got [%s].",
            dimC, op::ToString(weight->GetViewShape()).GetString());
        return false;
    }
    if (runningMean != nullptr &&
        (runningMean->GetViewShape().GetDimNum() != 1 || runningMean->GetViewShape()[0] != dimC)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim of runningMean should be one and shape is channel num of input[%d], but got [%s].", dimC,
            op::ToString(runningMean->GetViewShape()).GetString());
        return false;
    }
    if (runningVar != nullptr &&
        (runningVar->GetViewShape().GetDimNum() != 1 || runningVar->GetViewShape()[0] != dimC)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim of runningVar should be one and shape is channel num of input[%d], but got [%s].", dimC,
            op::ToString(runningVar->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static int64_t GetDimC(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    return viewShape[1];
}

static aclnnStatus CheckParams(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput, const aclTensor* gradWeight,
    const aclTensor* gradBias, const aclBoolArray* outputMask, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd)
{
    CHECK_RET(CheckMaskNotNull(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(gradOut, input), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGradDtypeValid(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOtherDtypeValid(weight, runningMean, runningVar, saveMean, saveInvstd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);
    int dimC = GetDimC(input);
    CHECK_RET(CheckOtherShape(dimC, weight, runningMean, runningVar), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormPost(
    op::Shape& inputShape, aclTensor* bnGradInput, aclTensor* gradInput, aclOpExecutor* executor)
{
    auto inputDims = inputShape.GetDimNum();
    if (inputDims > MAX_BN_DIMS) {
        int64_t originShapes[inputDims];
        for (size_t i = 0; i < inputDims; ++i) {
            originShapes[i] = inputShape[i];
        }
        aclIntArray* originShapeArray = executor->AllocIntArray(originShapes, inputDims);
        auto bnGradInputReshape = l0op::Reshape(bnGradInput, originShapeArray, executor);
        auto bnGradInputReformat = l0op::ReFormat(bnGradInputReshape, Format::FORMAT_ND);
        auto viewCopyResult = l0op::ViewCopy(bnGradInputReformat, gradInput, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto viewCopyResult = l0op::ViewCopy(bnGradInput, gradInput, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackwardProc(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias, aclOpExecutor* executor)
{
    bool isSupportNcdhw = isBatchNormSupportNcdhw();
    auto weightResize = op::ResizeFrom1D(weight, input, isSupportNcdhw, executor);
    CHECK_RET(weightResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::array<aclTensor*, UPDATE_GRAD_RESULT_CNT> grad;
    if (!training) {
        auto runningMeanResize = op::ResizeFrom1D(runningMean, input, isSupportNcdhw, executor);
        CHECK_RET(runningMeanResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto runningVarResize = op::ResizeFrom1D(runningVar, input, isSupportNcdhw, executor);
        CHECK_RET(runningVarResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        grad = l0op::BNTrainingUpdateGrad(gradOut, input, runningMeanResize, runningVarResize, eps, executor);

        auto reduceGrad = l0op::BNInferGrad(gradOut, weightResize, runningVarResize, eps, executor);
        CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor*>(reduceGrad);
    } else {
        auto saveMeanResize = op::ResizeFrom1D(saveMean, input, isSupportNcdhw, executor);
        CHECK_RET(saveMeanResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto saveInvstdResize = op::ResizeFrom1D(saveInvstd, input, isSupportNcdhw, executor);
        CHECK_RET(saveInvstdResize != nullptr, ACLNN_ERR_INNER_NULLPTR);

        if (input->GetViewShape().GetDimNum() == MAX_BN_DIMS) {
            grad = l0op::BN3DTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            auto reduceGrad = l0op::BN3DTrainingReduceGrad(
                gradOut, input, grad[0], grad[1], weightResize, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            *gradInput = const_cast<aclTensor*>(reduceGrad);
        } else {
            grad = l0op::BNTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            auto reduceGrad = l0op::BNTrainingReduceGrad(
                gradOut, input, grad[0], grad[1], weightResize, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            *gradInput = const_cast<aclTensor*>(reduceGrad);
        }
    }
    *gradWeight = grad[0];
    *gradBias = grad[1];
    return ACLNN_SUCCESS;
}

const aclTensor* Transpose(const aclTensor* input, aclOpExecutor* executor)
{
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();

    const int64_t value4d[] = {1, 0, 2, 3};
    const int64_t value5d[] = {1, 0, 2, 3, 4};

    aclIntArray* ndchwShape = executor->AllocIntArray((inputDim == 4) ? value4d : value5d, inputDim);
    auto inputTranspose = l0op::Transpose(input, ndchwShape, executor);
    if (inputTranspose == nullptr) {
        return inputTranspose;
    }
    // NDCHW -> NCHW
    const int64_t reshapeShape4d[] = {1, inputShape[1], inputShape[0], inputShape[2] * inputShape[3]};
    const int64_t reshapeShape5d[] = {1, inputShape[1], inputShape[0], inputShape[2], inputShape[3] * inputShape[4]};
    aclIntArray* reshapeArray;
    if (inputDim == BN2D_INPUT_DIMS) {
        reshapeArray = executor->AllocIntArray(reshapeShape4d, sizeof(reshapeShape4d) / sizeof(int64_t));
        auto inputReshape = l0op::Reshape(inputTranspose, reshapeArray, executor);
        if (inputReshape == nullptr) {
            return inputReshape;
        }
        return l0op::ReFormat(inputReshape, Format::FORMAT_NCHW);
    } else {
        reshapeArray = executor->AllocIntArray(reshapeShape5d, sizeof(reshapeShape5d) / sizeof(int64_t));
        auto inputReshape = l0op::Reshape(inputTranspose, reshapeArray, executor);
        if (inputReshape == nullptr) {
            return inputReshape;
        }
        return l0op::ReFormat(inputReshape, Format::FORMAT_NCDHW);
    }
}

const aclTensor* TransposeRevert(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor)
{
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();

    // nchw -> ndchw
    const int64_t orignShape4d[] = {inputShape[1], inputShape[0], inputShape[2], inputShape[3]};
    const int64_t orignShape5d[] = {inputShape[1], inputShape[0], inputShape[2], inputShape[3], inputShape[4]};
    aclIntArray* ndchwArray;
    if (inputDim == BN2D_INPUT_DIMS) {
        ndchwArray = executor->AllocIntArray(orignShape4d, sizeof(orignShape4d) / sizeof(int64_t));
    } else {
        ndchwArray = executor->AllocIntArray(orignShape5d, sizeof(orignShape5d) / sizeof(int64_t));
    }
    auto outputReshape = l0op::Reshape(output, ndchwArray, executor);
    if (outputReshape == nullptr) {
        return outputReshape;
    }

    auto outputFormat = l0op::ReFormat(outputReshape, inputDim == 4 ? Format::FORMAT_NCHW : Format::FORMAT_NCDHW);
    if (outputFormat == nullptr) {
        return outputFormat;
    }
    // ndchw -> ncdhw
    const int64_t ncdhwShape4d[] = {1, 0, 2, 3};
    const int64_t ncdhwShape5d[] = {1, 0, 2, 3, 4};
    aclIntArray* ncdhwArray = executor->AllocIntArray((inputDim == 4) ? ncdhwShape4d : ncdhwShape5d, inputDim);
    return l0op::Transpose(outputFormat, ncdhwArray, executor);
}

bool FastBatchNormBackwardProcTransposeCalc(
    const aclTensor* gradOut, const aclTensor** gradOutContiguous, const aclTensor** inputContiguous,
    aclOpExecutor* executor)
{
    auto gradOutShape = gradOut->GetViewShape();
    auto gradOutDims = gradOutShape.GetDimNum();

    size_t hwShape = 1;
    for (size_t i = 2; i < gradOutDims; i++) {
        hwShape *= gradOutShape[i];
    }
    if (gradOutShape[0] > BN_TRANSPOSE_N_LIMIT && gradOutShape[1] > 1 && hwShape <= BN_TRANSPOSE_HW_LIMIT) {
        *gradOutContiguous = Transpose(*gradOutContiguous, executor);
        CHECK_RET(*gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *inputContiguous = Transpose(*inputContiguous, executor);
        CHECK_RET(*inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return true;
    }
    return false;
}

aclnnStatus FastBatchNormBackwardProcForAscendC(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias, aclOpExecutor* executor)
{
    auto gradOutContiguous = l0op::Contiguous(gradOut, executor);
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool isTranspose = FastBatchNormBackwardProcTransposeCalc(gradOut, &gradOutContiguous, &inputContiguous, executor);

    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightCast = l0op::Cast(weightContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanContiguous = l0op::Contiguous(runningMean, executor);
    CHECK_RET(runningMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningMeanCast = l0op::Cast(runningMeanContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(runningMeanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarContiguous = l0op::Contiguous(runningVar, executor);
    CHECK_RET(runningVarContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarCast = l0op::Cast(runningVarContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(runningVarCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!training) {
        size_t dimC = input->GetViewShape()[1];
        saveMean = op::FillScalar(dimC, 0, executor);
    }

    auto saveMeanContiguous = l0op::Contiguous(saveMean, executor);
    CHECK_RET(saveMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto saveMeanCast = l0op::Cast(saveMeanContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(saveMeanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto saveInvstdContiguous = l0op::Contiguous(saveInvstd, executor);
    CHECK_RET(saveInvstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto saveInvstdCast = l0op::Cast(saveInvstdContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(saveInvstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::array<aclTensor*, BN_GRAD_V3_RESULT_CNT> grad = l0op::BatchNormGradV3(
        gradOutContiguous, inputContiguous, weightCast, runningMeanCast, runningVarCast, saveMeanCast, saveInvstdCast,
        training, eps, executor);
    *gradWeight = const_cast<aclTensor*>(grad[ONE]);
    *gradBias = const_cast<aclTensor*>(grad[TWO]);

    if (isTranspose) {
        auto test = TransposeRevert((grad[0]), input, executor);
        *gradInput = const_cast<aclTensor*>(test);
    } else {
        *gradInput = const_cast<aclTensor*>(grad[0]);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackwardPrepare(
    const aclTensor* input, const aclTensor*& weight, const aclTensor*& runningMean, const aclTensor*& runningVar,
    const aclTensor*& saveMean, const aclTensor*& saveInvstd, aclOpExecutor* executor)
{
    size_t dimC = input->GetViewShape()[1];
    if (runningMean == nullptr) {
        runningMean = op::FillScalar(dimC, 0, executor);
        CHECK_RET(runningMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (runningVar == nullptr) {
        runningVar = op::FillScalar(dimC, 1, executor);
        CHECK_RET(runningVar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (weight == nullptr) {
        weight = op::FillScalar(dimC, 1, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (saveMean == nullptr) {
        saveMean = op::FillScalar(dimC, 0, executor);
        CHECK_RET(saveMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (saveInvstd == nullptr) {
        saveInvstd = op::FillScalar(dimC, 1, executor);
        CHECK_RET(saveInvstd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackward(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias, aclOpExecutor* executor)
{
    aclnnStatus bnPrepareResp =
        FastBatchNormBackwardPrepare(input, weight, runningMean, runningVar, saveMean, saveInvstd, executor);
    CHECK_RET(bnPrepareResp == ACLNN_SUCCESS, bnPrepareResp);

    size_t dimNum = input->GetViewShape().GetDimNum();
    auto gradOutPre = gradOut;
    auto inputPre = input;
    if (dimNum < BN2D_INPUT_DIMS) {
        gradOutPre = op::ResizeFromND(gradOut, executor);
        CHECK_RET(gradOutPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        inputPre = op::ResizeFromND(input, executor);
        CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor* result = nullptr;
    if (isBatchNormSupportAscendC()) {
        auto bnResult = FastBatchNormBackwardProcForAscendC(
            gradOutPre, inputPre, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, &result,
            gradWeight, gradBias, executor);
        CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
    } else {
        auto bnResult = FastBatchNormBackwardProc(
            gradOutPre, inputPre, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, &result,
            gradWeight, gradBias, executor);
        CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
    }

    *gradInput = result;
    if (dimNum < BN2D_INPUT_DIMS) {
        auto outputFormat = op::ResizeToND(result, input, executor);
        CHECK_RET(outputFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor*>(outputFormat);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackwardWeightAndBiasPost(
    const aclBoolArray* outputMask, aclTensor* bnGradWeight, aclTensor* gradWeight, aclTensor* bnGradBias,
    aclTensor* gradBias, aclOpExecutor* executor)
{
    if (!isBatchNormSupportAscendC()) {
        if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
            auto viewCopyWeight =
                op::ResizeTo1D(bnGradWeight, gradWeight, isBatchNormSupportNcdhw(), executor);
            CHECK_RET(viewCopyWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[GRAD_BIAS_INDEX]) {
            auto viewCopyBias = op::ResizeTo1D(bnGradBias, gradBias, isBatchNormSupportNcdhw(), executor);
            CHECK_RET(viewCopyBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
            auto bnGradWeightCast = l0op::Cast(bnGradWeight, gradWeight->GetDataType(), executor);
            CHECK_RET(bnGradWeightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyWeight = l0op::ViewCopy(bnGradWeightCast, gradWeight, executor);
            CHECK_RET(viewCopyWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[GRAD_BIAS_INDEX]) {
            auto bnGradBiasCast = l0op::Cast(bnGradBias, gradBias->GetDataType(), executor);
            CHECK_RET(bnGradBiasCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyBias = l0op::ViewCopy(bnGradBiasCast, gradBias, executor);
            CHECK_RET(viewCopyBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, double eps,
    const aclBoolArray* outputMask, int version, aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnFastBatchNormBackward,
        DFX_IN(gradOut, input, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, outputMask, version),
        DFX_OUT(gradInput, gradWeight, gradBias));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckInputNotNull(gradOut, input), ACLNN_ERR_PARAM_NULLPTR);

    if (input->IsEmpty() || gradOut->IsEmpty()) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = CheckParams(
        gradOut, input, gradInput, gradWeight, gradBias, outputMask, weight, runningMean, runningVar, saveMean,
        saveInvstd);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutContigous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContigous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputShape = input->GetViewShape();
    if (inputShape.GetDimNum() > MAX_BN_DIMS) {
        const int64_t shapes[5] = {inputShape[0], inputShape[1], inputShape[2], inputShape[3], -1};
        aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(shapes, 5);
        inputContiguous = l0op::Reshape(inputContiguous, shapeArray, uniqueExecutor.get());
        inputContiguous = l0op::ReFormat(inputContiguous, Format::FORMAT_NCDHW);
        gradOutContigous = l0op::Reshape(gradOutContigous, shapeArray, uniqueExecutor.get());
        gradOutContigous = l0op::ReFormat(gradOutContigous, Format::FORMAT_NCDHW);
    }

    aclTensor* bnGradInput = nullptr;
    aclTensor* bnGradWeight = nullptr;
    aclTensor* bnGradBias = nullptr;
    aclnnStatus bnResult = FastBatchNormBackward(
        gradOutContigous, inputContiguous, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps,
        &bnGradInput, &bnGradWeight, &bnGradBias, uniqueExecutor.get());
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
    auto weightResult = FastBatchNormBackwardWeightAndBiasPost(outputMask, bnGradWeight, gradWeight, bnGradBias, gradBias, uniqueExecutor.get());
    CHECK_RET(weightResult == ACLNN_SUCCESS, weightResult);
    if ((*outputMask)[0]) {
        auto viewCopyInput = FastBatchNormPost(inputShape, bnGradInput, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyInput == ACLNN_SUCCESS, viewCopyInput);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastBatchNormBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFastBatchNormBackward);
    // 固定写法，调用框架能力，完成计算
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
