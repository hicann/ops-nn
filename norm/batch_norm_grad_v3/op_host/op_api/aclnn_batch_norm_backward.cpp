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
#include "aclnn_kernels/reshape.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
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
#include "op_api/aclnn_util.h"
#include "batch_norm_backward.h"
#include "aclnn_batch_norm_backward.h"

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
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return (
        (curArch == NpuArch::DAV_2201) ||
        (curArch == NpuArch::DAV_2002));
}

static bool CheckInputNotNull(const aclTensor* gradOut, const aclTensor* input)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(input, return false);

    return true;
}

static bool CheckMaskNotNull(
    const aclTensor* gradInput, const aclTensor* gradWeight, const aclTensor* gradBias, const aclBoolArray* outputMask)
{
    OP_CHECK_NULL(outputMask, return false);
    if (outputMask->Size() < GRAD_WEIGHT_INDEX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputMask size should not less than 1.");
        return false;
    }
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_NULL(outputMask, return false);
        if (outputMask->Size() != BN_GRAD_V3_RESULT_CNT) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "outputMask size should be %zu, but got %zu.", BN_GRAD_V3_RESULT_CNT,
                outputMask->Size());
            return false;
        }
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
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
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
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        if ((input->GetViewShape().GetDimNum() == MAX_BN_DIMS) &&
            ((input->GetStorageFormat() != Format::FORMAT_NCDHW) &&
             (input->GetStorageFormat() != Format::FORMAT_NDHWC))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input should be NCDWH or NDWHC, when input dim is 5.");
            return false;
        }

        if (((*outputMask)[0] && gradInput->GetViewShape().GetDimNum() == MAX_BN_DIMS) &&
            ((gradInput->GetStorageFormat() != Format::FORMAT_NCDHW) &&
             (gradInput->GetStorageFormat() != Format::FORMAT_NDHWC))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of output should be NCDWH or NDWHC, when input dim is 5.");
            return false;
        }
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

static bool CheckSaveMeanSaveInvstdShape(int dimC, const aclTensor* saveMean, const aclTensor* saveInvstd,
                                         bool training)
{
    if (saveMean != nullptr &&
        (saveMean->GetViewShape().GetDimNum() != 1 ||
         (saveMean->GetViewShape()[0] != dimC &&
          (training || saveMean->GetViewShape()[0] != 0)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim of saveMean should be one and shape is channel num of input[%d], but got [%s].", dimC,
            op::ToString(saveMean->GetViewShape()).GetString());
        return false;
    }
    if (saveInvstd != nullptr &&
        (saveInvstd->GetViewShape().GetDimNum() != 1 ||
         (saveInvstd->GetViewShape()[0] != dimC &&
          (training || saveInvstd->GetViewShape()[0] != 0)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dim of saveInvstd should be one and shape is channel num of input[%d], but got [%s].", dimC,
            op::ToString(saveInvstd->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckGradWeightGradBiasShape(
    int dimC, const aclTensor* gradWeight, const aclTensor* gradBias, const aclBoolArray* outputMask)
{
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        if (gradWeight != nullptr &&
            (gradWeight->GetViewShape().GetDimNum() != 1 || gradWeight->GetViewShape()[0] != dimC)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Dim of gradWeight should be one and shape is channel num of input[%d], but got [%s].", dimC,
                op::ToString(gradWeight->GetViewShape()).GetString());
            return false;
        }
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        if (gradBias != nullptr && (gradBias->GetViewShape().GetDimNum() != 1 || gradBias->GetViewShape()[0] != dimC)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Dim of gradBias should be one and shape is channel num of input[%d], but got [%s].", dimC,
                op::ToString(gradBias->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static int64_t GetDimC(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        if (input->GetStorageFormat() == Format::FORMAT_NDHWC || input->GetStorageFormat() == Format::FORMAT_NHWC) {
            return viewShape[viewShape.GetDimNum() - 1];
        }
    }
    return viewShape[1];
}

static bool CheckDtypeMatchReferenceOrFloat(
    const aclTensor* tensor, const aclTensor* reference, const char* tensorName, const char* referenceName)
{
    if (tensor == nullptr || reference == nullptr) {
        return true;
    }

    if (tensor->GetDataType() != reference->GetDataType() && tensor->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dtype of %s:[%s] should be float or same as %s, %s dtype: [%s].", tensorName,
            op::ToString(tensor->GetDataType()).GetString(), referenceName, referenceName,
            op::ToString(reference->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool DavidCheckDtypeSame(
    const aclTensor* gradOut, const aclTensor* gradInput, const aclTensor* gradWeight, const aclTensor* gradBias,
    const aclBoolArray* outputMask, const aclTensor* weight, const aclTensor* runningMean, const aclTensor* runningVar,
    const aclTensor* saveMean, const aclTensor* saveInvstd)
{
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(weight, gradOut, "weight", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(runningMean, gradOut, "runningMean", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(runningVar, gradOut, "runningVar", "gradOut"), false);

    // runningMean 和 runningVar 类型一致
    if (runningMean != nullptr && runningVar != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(runningMean, runningVar, return false);
    }

    CHECK_RET(CheckDtypeMatchReferenceOrFloat(saveMean, gradOut, "saveMean", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(saveInvstd, gradOut, "saveInvstd", "gradOut"), false);

    // saveInvstd 和 saveMean 类型一致
    if (saveInvstd != nullptr && saveMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(saveInvstd, saveMean, return false);
    }

    // gradInput 和 gradOut 类型一致
    if ((*outputMask)[0]) {
        OP_CHECK_DTYPE_NOT_SAME(gradInput, gradOut, return false);
    }
    // gradWeight 和 weight 类型一致或者float
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        CHECK_RET(CheckDtypeMatchReferenceOrFloat(gradWeight, weight, "gradWeight", "weight"), false);
    }
    // gradBias 和 weight 类型一致或者float
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        CHECK_RET(CheckDtypeMatchReferenceOrFloat(gradBias, weight, "gradBias", "weight"), false);
    }

    if (gradWeight != nullptr && gradBias != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(gradWeight, gradBias, return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput, const aclTensor* gradWeight,
    const aclTensor* gradBias, const aclBoolArray* outputMask, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training)
{
    CHECK_RET(CheckMaskNotNull(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(gradOut, input), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGradDtypeValid(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOtherDtypeValid(weight, runningMean, runningVar, saveMean, saveInvstd), ACLNN_ERR_PARAM_INVALID);

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        CHECK_RET(
            DavidCheckDtypeSame(
                gradOut, gradInput, gradWeight, gradBias, outputMask, weight, runningMean, runningVar, saveMean,
                saveInvstd),
            ACLNN_ERR_PARAM_INVALID);
    }

    CHECK_RET(CheckFormat(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);
    int dimC = GetDimC(input);
    CHECK_RET(CheckOtherShape(dimC, weight, runningMean, runningVar), ACLNN_ERR_PARAM_INVALID);

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        // 校验saveMean和saveInvstd的shape
        CHECK_RET(CheckSaveMeanSaveInvstdShape(dimC, saveMean, saveInvstd, training), ACLNN_ERR_PARAM_INVALID);
        // 校验gradWeight和gradBias的shape
        CHECK_RET(CheckGradWeightGradBiasShape(dimC, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormPost(op::Shape& inputShape, aclTensor* bnGradInput, aclTensor* gradInput, aclOpExecutor* executor)
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

aclnnStatus BatchNormBackwardProc(
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

        if (input->GetViewShape().GetDimNum() == MAX_BN_DIMS && !isBatchNormSupportNcdhw()) {
            auto gradOut6hd = l0op::TransDataSpecial(gradOut, Format::FORMAT_NDC1HWC0, 0, executor);
            CHECK_RET(gradOut6hd != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto input6hd = l0op::TransDataSpecial(input, Format::FORMAT_NDC1HWC0, 0, executor);
            CHECK_RET(input6hd != nullptr, ACLNN_ERR_INNER_NULLPTR);

            grad = l0op::BN3DTrainingUpdateGrad(gradOut6hd, input6hd, saveMeanResize, saveInvstdResize, eps, executor);
            auto reduceGrad = l0op::BN3DTrainingReduceGrad(
                gradOut6hd, input6hd, grad[0], grad[1], weightResize, saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto resultNcdhw = l0op::TransDataSpecial(reduceGrad, Format::FORMAT_NCDHW, 0, executor);
            CHECK_RET(resultNcdhw != nullptr, ACLNN_ERR_INNER_NULLPTR);
            *gradInput = const_cast<aclTensor*>(resultNcdhw);
        } else if (input->GetViewShape().GetDimNum() == MAX_BN_DIMS) {
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

aclnnStatus BatchNormBackward(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias, aclOpExecutor* executor)
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

    size_t dimNum = input->GetViewShape().GetDimNum();
    auto gradOutPre = gradOut;
    auto inputPre = input;
    if (dimNum < BN2D_INPUT_DIMS) {
        gradOutPre = op::ResizeFromND(gradOut, executor);
        CHECK_RET(gradOutPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        inputPre = op::ResizeFromND(input, executor);
        CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (!training && dimNum == MAX_BN_DIMS) {
        gradOutPre = op::ResizeFrom5D(gradOut, executor);
        CHECK_RET(gradOutPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        inputPre = op::ResizeFrom5D(input, executor);
        CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor* result = nullptr;
    auto bnResult = BatchNormBackwardProc(
        gradOutPre, inputPre, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, &result, gradWeight,
        gradBias, executor);
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    *gradInput = result;
    if (dimNum < BN2D_INPUT_DIMS) {
        auto outputFormat = op::ResizeToND(result, input, executor);
        CHECK_RET(outputFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor*>(outputFormat);
    } else if (!training && dimNum == MAX_BN_DIMS) {
        auto outputTranspose = op::ResizeTo5D(result, input, executor);
        CHECK_RET(outputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor*>(outputTranspose);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeGradWeight(
    const aclIntArray* reduceDim, const aclTensor* gradOutCasted, const aclTensor* inputCasted,
    const aclTensor* meanCasted, const aclTensor* varCasted, const float eps, aclTensor** gradWeight,
    aclOpExecutor* executor)
{
    // compute invstd
    auto epsScalar = executor->AllocScalar(static_cast<float>(eps));
    CHECK_RET(epsScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto epsTensor = executor->ConvertToTensor(epsScalar, op::DataType::DT_FLOAT);
    CHECK_RET(epsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto var1 = l0op::Add(varCasted, epsTensor, executor);
    CHECK_RET(var1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto expScalar = executor->AllocScalar(EXP_INV_SQRT);
    CHECK_RET(expScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invSqrtExp = executor->ConvertToTensor(expScalar, op::DataType::DT_FLOAT);
    CHECK_RET(invSqrtExp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstd = l0op::Pow(var1, invSqrtExp, executor);
    CHECK_RET(invstd != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // unsqueeze
    auto unsqueezeMean = meanCasted;
    auto unsqueezeInvstd = invstd;
    if (inputCasted->GetStorageFormat() == Format::FORMAT_NCDHW ||
        inputCasted->GetStorageFormat() == Format::FORMAT_NCHW) {
        uint64_t repeatCnt = inputCasted->GetViewShape().GetDimNum() - 2;
        for (uint64_t i = 0; i < repeatCnt; i++) {
            unsqueezeMean = l0op::UnsqueezeNd(unsqueezeMean, -1, executor);
            CHECK_RET(unsqueezeMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
            unsqueezeInvstd = l0op::UnsqueezeNd(unsqueezeInvstd, -1, executor);
            CHECK_RET(unsqueezeInvstd != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // compute grad weight
    auto xHat = l0op::Sub(inputCasted, unsqueezeMean, executor);
    CHECK_RET(xHat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xHatMul = l0op::Mul(xHat, unsqueezeInvstd, executor);
    CHECK_RET(xHatMul != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mulRet = l0op::Mul(gradOutCasted, xHatMul, executor);
    CHECK_RET(mulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto reduceSumDyXRet = l0op::ReduceSumOp(mulRet, reduceDim, false, executor);
    CHECK_RET(reduceSumDyXRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *gradWeight = const_cast<aclTensor*>(reduceSumDyXRet);

    return ACLNN_SUCCESS;
}

static aclnnStatus CalcGradWeightGradBias(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* runningMean, const aclTensor* runningVar,
    const float eps, const aclBoolArray* outputMask, aclTensor** gradWeight, aclTensor** gradBias,
    aclOpExecutor* executor)
{
    if (!(*outputMask)[GRAD_WEIGHT_INDEX] && !(*outputMask)[GRAD_BIAS_INDEX]) {
        return ACLNN_SUCCESS;
    }

    // cast
    auto gradOutCasted = l0op::Cast(gradOut, op::DataType::DT_FLOAT, executor);
    CHECK_RET(gradOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputCasted = l0op::Cast(input, op::DataType::DT_FLOAT, executor);
    CHECK_RET(inputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanCasted = l0op::Cast(runningMean, op::DataType::DT_FLOAT, executor);
    CHECK_RET(meanCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto varCasted = l0op::Cast(runningVar, op::DataType::DT_FLOAT, executor);
    CHECK_RET(varCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // reduce轴计算
    size_t inputDim = inputCasted->GetViewShape().GetDimNum();
    uint64_t dimCIdx = inputDim - 1UL;
    if (inputCasted->GetStorageFormat() == Format::FORMAT_NCDHW ||
        inputCasted->GetStorageFormat() == Format::FORMAT_NCHW) {
        dimCIdx = 1UL;
    }
    int64_t appendDim[inputDim - 1UL];
    uint64_t dimIdx = 0;
    for (uint64_t i = 0; i < inputDim; i++) {
        if (i == dimCIdx) {
            continue;
        }
        appendDim[dimIdx++] = static_cast<int64_t>(i);
    }
    aclIntArray* dim = executor->AllocIntArray(appendDim, inputDim - 1);

    // grad weight计算
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        auto ret = ComputeGradWeight(dim, gradOutCasted, inputCasted, meanCasted, varCasted, eps, gradWeight, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }

    // grad bias计算
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        auto reduceSumDyRet = l0op::ReduceSumOp(gradOutCasted, dim, false, executor);
        CHECK_RET(reduceSumDyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *gradBias = const_cast<aclTensor*>(reduceSumDyRet);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormBackwardProcDavid(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    const aclBoolArray* outputMask, aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias,
    aclOpExecutor* executor)
{
    auto gradOutContiguous = l0op::Contiguous(gradOut, executor);
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (gradOutContiguous->GetDataType() == op::DataType::DT_FLOAT) {
        weightContiguous = l0op::Cast(weightContiguous, op::DataType::DT_FLOAT, executor);
        CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto runningMeanContiguous = l0op::Contiguous(runningMean, executor);
    CHECK_RET(runningMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto runningVarContiguous = l0op::Contiguous(runningVar, executor);
    CHECK_RET(runningVarContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (gradOutContiguous->GetDataType() == op::DataType::DT_FLOAT ||
            weightContiguous->GetDataType() == op::DataType::DT_FLOAT) {
        runningMeanContiguous = l0op::Cast(runningMeanContiguous, op::DataType::DT_FLOAT, executor);
        CHECK_RET(runningMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        runningVarContiguous = l0op::Cast(runningVarContiguous, op::DataType::DT_FLOAT, executor);
        CHECK_RET(runningVarContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto saveMeanContiguous = l0op::Contiguous(saveMean, executor);
    CHECK_RET(saveMeanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (saveMeanContiguous->IsEmpty()) {
        auto saveMeanTmp = const_cast<aclTensor*>(saveMeanContiguous);
        saveMeanTmp->SetDataType(op::DataType::DT_FLOAT);
        saveMeanContiguous = saveMeanTmp;
    }
    auto saveMeanCast = l0op::Cast(saveMeanContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(saveMeanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto saveInvstdContiguous = l0op::Contiguous(saveInvstd, executor);
    CHECK_RET(saveInvstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (saveInvstdContiguous->IsEmpty()) {
        auto saveInvstdTmp = const_cast<aclTensor*>(saveInvstdContiguous);
        saveInvstdTmp->SetDataType(op::DataType::DT_FLOAT);
        saveInvstdContiguous = saveInvstdTmp;
    }
    auto saveInvstdCast = l0op::Cast(saveInvstdContiguous, op::DataType::DT_FLOAT, executor);
    CHECK_RET(saveInvstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::array<aclTensor*, BN_GRAD_V3_RESULT_CNT> grad = l0op::BatchNormGradV3(
        gradOutContiguous, inputContiguous, weightContiguous, runningMeanContiguous, runningVarContiguous, saveMeanCast,
        saveInvstdCast, training, eps, executor);

    if (!training) {
        CHECK_RET(
            CalcGradWeightGradBias(
                gradOutContiguous, inputContiguous, runningMeanContiguous, runningVarContiguous, eps, outputMask,
                gradWeight, gradBias, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
        *gradInput = grad[0];
        return ACLNN_SUCCESS;
    }

    size_t idx = 0;
    *gradInput = grad[idx++];
    *gradWeight = grad[idx++];
    *gradBias = grad[idx++];

    return ACLNN_SUCCESS;
}

aclnnStatus BatchNormBackwardDavid(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
    const aclBoolArray* outputMask, aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias,
    aclOpExecutor* executor)
{
    size_t dimC = GetDimC(input);
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
    auto bnResult = BatchNormBackwardProcDavid(
        gradOutPre, inputPre, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, outputMask, &result,
        gradWeight, gradBias, executor);
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    *gradInput = result;
    if ((*outputMask)[0] && dimNum < BN2D_INPUT_DIMS) {
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto outputFormat = op::ResizeToND(result, input, executor);
        CHECK_RET(outputFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *gradInput = const_cast<aclTensor*>(outputFormat);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight, const aclTensor* runningMean,
    const aclTensor* runningVar, const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, double eps,
    const aclBoolArray* outputMask, aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnBatchNormBackward,
        DFX_IN(gradOut, input, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, outputMask),
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
        saveInvstd, training);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool canEarlyReturn = Ops::NN::AclnnUtil::IsRegbase() &&
      !(*outputMask)[0] && !(*outputMask)[GRAD_WEIGHT_INDEX] && !(*outputMask)[GRAD_BIAS_INDEX];
    if (canEarlyReturn) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

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
    aclnnStatus bnResult;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        bnResult = BatchNormBackwardDavid(
            gradOutContigous, inputContiguous, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps,
            outputMask, &bnGradInput, &bnGradWeight, &bnGradBias, uniqueExecutor.get());
        CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

        if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
            CHECK_RET(bnGradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gradWeightCast = l0op::Cast(bnGradWeight, gradWeight->GetDataType(), uniqueExecutor.get());
            CHECK_RET(gradWeightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gradWeightResult = l0op::ViewCopy(gradWeightCast, gradWeight, uniqueExecutor.get());
            CHECK_RET(gradWeightResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[GRAD_BIAS_INDEX]) {
            CHECK_RET(bnGradBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gradBiasCast = l0op::Cast(bnGradBias, gradBias->GetDataType(), uniqueExecutor.get());
            CHECK_RET(gradBiasCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto gradBiasResult = l0op::ViewCopy(gradBiasCast, gradBias, uniqueExecutor.get());
            CHECK_RET(gradBiasResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[0]) {
            CHECK_RET(bnGradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyInput = BatchNormPost(inputShape, bnGradInput, gradInput, uniqueExecutor.get());
            CHECK_RET(viewCopyInput == ACLNN_SUCCESS, viewCopyInput);
        }
    } else {
        bnResult = BatchNormBackward(
            gradOutContigous, inputContiguous, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps,
            &bnGradInput, &bnGradWeight, &bnGradBias, uniqueExecutor.get());
        CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
        if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
            auto viewCopyWeight =
                op::ResizeTo1D(bnGradWeight, gradWeight, isBatchNormSupportNcdhw(), uniqueExecutor.get());
            CHECK_RET(viewCopyWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[GRAD_BIAS_INDEX]) {
            auto viewCopyBias = op::ResizeTo1D(bnGradBias, gradBias, isBatchNormSupportNcdhw(), uniqueExecutor.get());
            CHECK_RET(viewCopyBias != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        if ((*outputMask)[0]) {
            auto viewCopyInput = BatchNormPost(inputShape, bnGradInput, gradInput, uniqueExecutor.get());
            CHECK_RET(viewCopyInput == ACLNN_SUCCESS, viewCopyInput);
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
