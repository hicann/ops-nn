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
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "level0/mul.h"
#include "level0/reduce_sum_op.h"
#include "level0/add.h"
#include "level0/sub.h"
#include "level0/pow.h"
#include "level0/div.h"
#include "level0/unsqueeze.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "norm/common/op_api/batch_norm_backward.h"
#include "aclnn_fast_batch_norm_backward.h"
#include "log/log.h"

using namespace op;

namespace {
// aclnn op_api层空指针失败出口统一上报EZ1011(nnopbase注册码, Arglist={functionName,paraName})。
// op_api编译单元带-DOP_LOG_LIBOPAPI_ONLY, log/log.h内OP_LOGE_WITHOUT_REPORT(opName版)不可见,
// 日志走始终可见的OP_LOGE_LIBOPAPI_REPORT。
void ReportNullPointerError(const char* functionName, const char* paraName)
{
    std::string funcName = functionName;
    std::string paramName = paraName;
    std::string msg = funcName + " failed because " + paramName + " cannot be a NULL pointer.";
    OP_LOGE_LIBOPAPI_REPORT(funcName.c_str(), "%s", msg.c_str());
    const std::vector<const char*> msgKey = {"functionName", "paraName"};
    const std::vector<const char*> msgValue = {funcName.c_str(), paramName.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1011", msgKey, msgValue);
}
} // namespace
#ifdef __cplusplus
extern "C" {
#endif

// batch norm backward constants
constexpr size_t GRAD_WEIGHT_INDEX = 1;
constexpr size_t GRAD_BIAS_INDEX = 2;
constexpr size_t FAST_BN_GRAD_RESULT_CNT = 3;
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
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool isBatchNormSupportNcdhw(void)
{
    return ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93));
}

static bool CheckInputNotNull(const aclTensor* gradOut, const aclTensor* input)
{
    OP_CHECK(gradOut != nullptr, ReportNullPointerError(__func__, "gradOut"), return false);
    OP_CHECK(input != nullptr, ReportNullPointerError(__func__, "input"), return false);

    return true;
}

static inline bool isBatchNormSupportAscendC(void)
{
    bool socVersionSupport = ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) ||
                              (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93));

    return socVersionSupport;
}

// outputMask长度必须为3，且需在访问mask[0..2]之前完成校验，防止越界读（issue#7080）。
// null返回161001，长度非法返回161002，与文档错误码表一致。
static aclnnStatus CheckOutputMask(const aclBoolArray* fastOutputMask)
{
    OP_CHECK(fastOutputMask != nullptr, ReportNullPointerError(__func__, "fastOutputMask"),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (fastOutputMask->Size() != FAST_BN_GRAD_RESULT_CNT) {
        OP_LOGE_FOR_INVALID_LISTSIZE("aclnnFastBatchNormBackward", "fastOutputMask",
                                     std::to_string(fastOutputMask->Size()).c_str(),
                                     std::to_string(FAST_BN_GRAD_RESULT_CNT).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool CheckMaskNotNull(const aclTensor* fastGradInput, const aclTensor* fastGradWeight,
                             const aclTensor* fastGradBias, const aclBoolArray* fastOutputMask)
{
    // fast batch norm backward: check output mask validity
    OP_CHECK(fastOutputMask != nullptr, ReportNullPointerError(__func__, "fastOutputMask"), return false);
    if ((*fastOutputMask)[0]) {
        OP_CHECK(fastGradInput != nullptr, ReportNullPointerError(__func__, "fastGradInput"), return false);
    }
    if ((*fastOutputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK(fastGradWeight != nullptr, ReportNullPointerError(__func__, "fastGradWeight"), return false);
    }
    if ((*fastOutputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK(fastGradBias != nullptr, ReportNullPointerError(__func__, "fastGradBias"), return false);
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
    const auto& fastSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, fastSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, fastSupportList, return false);
    OP_CHECK_DTYPE_NOT_SAME(input, gradOut, return false);

    return true;
}

static bool CheckGradDtypeValid(const aclTensor* gradInput, const aclTensor* gradWeight, const aclTensor* gradBias,
                                const aclBoolArray* outputMask)
{
    const auto& fastSupportList = GetDtypeSupportList();
    if ((*outputMask)[0]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, fastSupportList, return false);
    }
    if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradWeight, fastSupportList, return false);
    }
    if ((*outputMask)[GRAD_BIAS_INDEX]) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gradBias, fastSupportList, return false);
    }

    return true;
}

static bool CheckOtherDtypeValid(const aclTensor* weight, const aclTensor* runningMean, const aclTensor* runningVar,
                                 const aclTensor* saveMean, const aclTensor* saveInvstd)
{
    const auto& fastSupportList = GetDtypeSupportList();
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weight, fastSupportList, return false);
    }
    if (runningMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningMean, fastSupportList, return false);
    }
    if (runningVar != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(runningVar, fastSupportList, return false);
    }
    if (saveMean != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(saveMean, fastSupportList, return false);
    }
    if (saveInvstd != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(saveInvstd, fastSupportList, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput,
                        const aclBoolArray* outputMask)
{
    if (gradOut->GetStorageFormat() != input->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnFastBatchNormBackward", "gradOut,input",
                                                (std::string(op::ToString(gradOut->GetStorageFormat()).GetString()) +
                                                 ", " + op::ToString(input->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "format of gradOut and input should be equal");
        return false;
    }
    if ((*outputMask)[0] && gradOut->GetStorageFormat() != gradInput->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnFastBatchNormBackward", "gradOut,gradInput",
                                                (std::string(op::ToString(gradOut->GetStorageFormat()).GetString()) +
                                                 ", " + op::ToString(gradInput->GetStorageFormat()).GetString())
                                                    .c_str(),
                                                "format of gradOut and gradInput should be equal");
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOut->GetStorageFormat()) || op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnFastBatchNormBackward", "gradOut",
                                   op::ToString(gradOut->GetStorageFormat()).GetString(), "NC, NCL, NCHW or NCDHW");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput,
                       const aclBoolArray* outputMask)
{
    const int max_check_nums = 8;
    OP_CHECK_MAX_DIM(input, max_check_nums, return false);
    OP_CHECK_MIN_DIM(input, MIN_BN_DIMS, return false);
    OP_CHECK_MAX_DIM(gradOut, max_check_nums, return false);
    OP_CHECK_MIN_DIM(gradOut, MIN_BN_DIMS, return false);
    OP_CHECK(gradOut->GetViewShape() == input->GetViewShape(),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnFastBatchNormBackward", "gradOut,input",
                                                    (std::string(op::ToString(gradOut->GetViewShape()).GetString()) +
                                                     ", " + op::ToString(input->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shapes of gradOut and input should be the same"),
             return false);

    if ((*outputMask)[0]) {
        OP_CHECK_MAX_DIM(gradInput, max_check_nums, return false);
        OP_CHECK_MIN_DIM(gradInput, MIN_BN_DIMS, return false);
        OP_CHECK(
            gradInput->GetViewShape() == gradOut->GetViewShape(),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnFastBatchNormBackward", "gradInput,gradOut",
                                                   (std::string(op::ToString(gradInput->GetViewShape()).GetString()) +
                                                    ", " + op::ToString(gradOut->GetViewShape()).GetString())
                                                       .c_str(),
                                                   "shapes of gradInput and gradOut should be the same"),
            return false);
    }
    return true;
}

static bool CheckOtherShape(bool training, int64_t fastDimC, const aclTensor* fastWeight,
                            const aclTensor* fastRunningMean, const aclTensor* fastRunningVar,
                            const aclTensor* fastSaveMean, const aclTensor* fastSaveInvstd,
                            const aclTensor* fastGradWeight, const aclTensor* fastGradBias,
                            const aclBoolArray* fastOutputMask)
{
    if (fastWeight != nullptr &&
        (fastWeight->GetViewShape().GetDimNum() != 1 || fastWeight->GetViewShape()[0] != fastDimC)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnFastBatchNormBackward", "fastWeight", op::ToString(fastWeight->GetViewShape()).GetString(),
            (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
        return false;
    }
    if (fastRunningMean != nullptr &&
        (fastRunningMean->GetViewShape().GetDimNum() != 1 || fastRunningMean->GetViewShape()[0] != fastDimC)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnFastBatchNormBackward", "fastRunningMean", op::ToString(fastRunningMean->GetViewShape()).GetString(),
            (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
        return false;
    }
    if (fastRunningVar != nullptr &&
        (fastRunningVar->GetViewShape().GetDimNum() != 1 || fastRunningVar->GetViewShape()[0] != fastDimC)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnFastBatchNormBackward", "fastRunningVar", op::ToString(fastRunningVar->GetViewShape()).GetString(),
            (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
        return false;
    }
    // 1D辅助tensor长度非C时原样流入内核会泄漏内部错误码，需在此按文档返回161002（issue#7080）。
    // 推理态不使用这两个张量（非AscendC路径忽略、AscendC路径由FillScalar覆盖），故仅在训练态校验。
    if (training) {
        if (fastSaveMean != nullptr &&
            (fastSaveMean->GetViewShape().GetDimNum() != 1 || fastSaveMean->GetViewShape()[0] != fastDimC)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnFastBatchNormBackward", "fastSaveMean", op::ToString(fastSaveMean->GetViewShape()).GetString(),
                (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
            return false;
        }
        if (fastSaveInvstd != nullptr &&
            (fastSaveInvstd->GetViewShape().GetDimNum() != 1 || fastSaveInvstd->GetViewShape()[0] != fastDimC)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnFastBatchNormBackward", "fastSaveInvstd",
                op::ToString(fastSaveInvstd->GetViewShape()).GetString(),
                (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
            return false;
        }
    }
    if ((*fastOutputMask)[GRAD_WEIGHT_INDEX] && fastGradWeight != nullptr &&
        (fastGradWeight->GetViewShape().GetDimNum() != 1 || fastGradWeight->GetViewShape()[0] != fastDimC)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnFastBatchNormBackward", "fastGradWeight", op::ToString(fastGradWeight->GetViewShape()).GetString(),
            (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
        return false;
    }
    if ((*fastOutputMask)[GRAD_BIAS_INDEX] && fastGradBias != nullptr &&
        (fastGradBias->GetViewShape().GetDimNum() != 1 || fastGradBias->GetViewShape()[0] != fastDimC)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnFastBatchNormBackward", "fastGradBias", op::ToString(fastGradBias->GetViewShape()).GetString(),
            (std::string("shape should be [") + std::to_string(fastDimC) + "]").c_str());
        return false;
    }
    return true;
}

// 辅助输入dtype口径与同族 aclnnBatchNormBackward 的 CheckDtypeMatchReferenceOrFloat 对齐：
// 与 gradOut 同型或 DT_FLOAT 均可（AscendC路径已将辅助输入Cast到FLOAT，fp32辅助张量为框架AMP
// 常规传参）；gradInput经ViewCopy写出不支持混型，保持与gradOut强一致（issue#7080）。
static bool CheckDtypeMatchReferenceOrFloat(const aclTensor* tensor, const aclTensor* reference, const char* tensorName,
                                            const char* referenceName)
{
    if (tensor == nullptr || reference == nullptr) {
        return true;
    }
    if (tensor->GetDataType() != reference->GetDataType() && tensor->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnFastBatchNormBackward", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            (std::string("dtype should be float or the same as ") + referenceName).c_str());
        return false;
    }
    return true;
}

static bool CheckAuxDtypeSame(const aclTensor* gradOut, const aclTensor* gradInput, const aclTensor* weight,
                              const aclTensor* runningMean, const aclTensor* runningVar, const aclTensor* saveMean,
                              const aclTensor* saveInvstd, const aclBoolArray* outputMask)
{
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(weight, gradOut, "weight", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(runningMean, gradOut, "runningMean", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(runningVar, gradOut, "runningVar", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(saveMean, gradOut, "saveMean", "gradOut"), false);
    CHECK_RET(CheckDtypeMatchReferenceOrFloat(saveInvstd, gradOut, "saveInvstd", "gradOut"), false);
    if ((*outputMask)[0]) {
        OP_CHECK_DTYPE_NOT_SAME(gradInput, gradOut, return false);
    }
    return true;
}

static int64_t GetDimC(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    return viewShape[1];
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput,
                               const aclTensor* gradWeight, const aclTensor* gradBias, const aclBoolArray* outputMask,
                               const aclTensor* weight, const aclTensor* runningMean, const aclTensor* runningVar,
                               const aclTensor* saveMean, const aclTensor* saveInvstd, bool training)
{
    auto maskRet = CheckOutputMask(outputMask);
    CHECK_RET(maskRet == ACLNN_SUCCESS, maskRet);

    CHECK_RET(CheckMaskNotNull(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(gradOut, input), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGradDtypeValid(gradInput, gradWeight, gradBias, outputMask), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOtherDtypeValid(weight, runningMean, runningVar, saveMean, saveInvstd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAuxDtypeSame(gradOut, gradInput, weight, runningMean, runningVar, saveMean, saveInvstd, outputMask),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(gradOut, input, gradInput, outputMask), ACLNN_ERR_PARAM_INVALID);
    int64_t dimC = GetDimC(input);
    CHECK_RET(CheckOtherShape(training, dimC, weight, runningMean, runningVar, saveMean, saveInvstd, gradWeight,
                              gradBias, outputMask),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormPost(op::Shape& inputShape, aclTensor* bnGradInput, aclTensor* gradInput,
                              aclOpExecutor* executor)
{
    auto inputDims = inputShape.GetDimNum();
    if (inputDims > MAX_BN_DIMS) {
        int64_t fastOriginShapes[inputDims];
        for (size_t i = 0; i < inputDims; ++i) {
            fastOriginShapes[i] = inputShape[i];
        }
        aclIntArray* fastOriginShapeArray = executor->AllocIntArray(fastOriginShapes, inputDims);
        auto bnGradInputReshape = l0op::Reshape(bnGradInput, fastOriginShapeArray, executor);
        auto bnGradInputReformat = l0op::ReFormat(bnGradInputReshape, Format::FORMAT_ND);
        auto viewCopyResult = l0op::ViewCopy(bnGradInputReformat, gradInput, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 4D/5D ND输入经ReFormat为NCHW/NCDHW计算后，写回前需对齐回用户的原始格式，
        // ViewCopy要求源与目标格式一致（issue#7080）
        auto bnGradInputAligned = (const aclTensor*)bnGradInput;
        if (bnGradInput->GetStorageFormat() != gradInput->GetStorageFormat()) {
            bnGradInputAligned = l0op::ReFormat(bnGradInput, gradInput->GetStorageFormat());
            CHECK_RET(bnGradInputAligned != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        auto viewCopyResult = l0op::ViewCopy(bnGradInputAligned, gradInput, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackwardProc(const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight,
                                      const aclTensor* runningMean, const aclTensor* runningVar,
                                      const aclTensor* saveMean, const aclTensor* saveInvstd, bool training, float eps,
                                      aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias,
                                      aclOpExecutor* executor)
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
            // fast batch norm: 3D training update grad
            grad = l0op::BN3DTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            auto reduceGrad = l0op::BN3DTrainingReduceGrad(gradOut, input, grad[0], grad[1], weightResize,
                                                           saveMeanResize, saveInvstdResize, eps, executor);
            CHECK_RET(reduceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);
            *gradInput = const_cast<aclTensor*>(reduceGrad);
        } else {
            // fast batch norm: 2D training update grad
            grad = l0op::BNTrainingUpdateGrad(gradOut, input, saveMeanResize, saveInvstdResize, eps, executor);
            auto reduceGrad = l0op::BNTrainingReduceGrad(gradOut, input, grad[0], grad[1], weightResize, saveMeanResize,
                                                         saveInvstdResize, eps, executor);
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

    aclIntArray* ndchwArray;
    if (inputDim == BN2D_INPUT_DIMS) {
        const int64_t orignShape4d[] = {inputShape[1], inputShape[0], inputShape[2], inputShape[3]};
        ndchwArray = executor->AllocIntArray(orignShape4d, sizeof(orignShape4d) / sizeof(int64_t));
    } else {
        const int64_t orignShape5d[] = {inputShape[1], inputShape[0], inputShape[2], inputShape[3], inputShape[4]};
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

bool FastBatchNormBackwardProcTransposeCalc(const aclTensor* gradOut, const aclTensor** gradOutContiguous,
                                            const aclTensor** inputContiguous, aclOpExecutor* executor)
{
    auto gradOutShape = gradOut->GetViewShape();
    auto gradOutDims = gradOutShape.GetDimNum();

    size_t hwShape = 1;
    for (size_t i = 2; i < gradOutDims; i++) {
        hwShape *= gradOutShape[i];
    }
    if (static_cast<size_t>(gradOutShape[0]) > BN_TRANSPOSE_N_LIMIT && gradOutShape[1] > 1 &&
        hwShape <= BN_TRANSPOSE_HW_LIMIT) {
        *gradOutContiguous = Transpose(*gradOutContiguous, executor);
        CHECK_RET(*gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *inputContiguous = Transpose(*inputContiguous, executor);
        CHECK_RET(*inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return true;
    }
    return false;
}

aclnnStatus FastBatchNormBackwardProcForAscendC(const aclTensor* gradOut, const aclTensor* input,
                                                const aclTensor* weight, const aclTensor* runningMean,
                                                const aclTensor* runningVar, const aclTensor* saveMean,
                                                const aclTensor* saveInvstd, bool training, float eps,
                                                aclTensor** gradInput, aclTensor** gradWeight, aclTensor** gradBias,
                                                aclOpExecutor* executor)
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

aclnnStatus FastBatchNormBackwardPrepare(const aclTensor* input, const aclTensor*& fastWeight,
                                         const aclTensor*& fastRunningMean, const aclTensor*& fastRunningVar,
                                         const aclTensor*& fastSaveMean, const aclTensor*& fastSaveInvstd,
                                         aclOpExecutor* executor)
{
    // fast batch norm backward: prepare default values for optional tensors
    size_t fastDimC = input->GetViewShape()[1];
    if (fastRunningMean == nullptr) {
        fastRunningMean = op::FillScalar(fastDimC, 0, executor);
        CHECK_RET(fastRunningMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (fastRunningVar == nullptr) {
        fastRunningVar = op::FillScalar(fastDimC, 1, executor);
        CHECK_RET(fastRunningVar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (fastWeight == nullptr) {
        fastWeight = op::FillScalar(fastDimC, 1, executor);
        CHECK_RET(fastWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (fastSaveMean == nullptr) {
        fastSaveMean = op::FillScalar(fastDimC, 0, executor);
        CHECK_RET(fastSaveMean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (fastSaveInvstd == nullptr) {
        fastSaveInvstd = op::FillScalar(fastDimC, 1, executor);
        CHECK_RET(fastSaveInvstd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus FastBatchNormBackward(const aclTensor* gradOut, const aclTensor* input, const aclTensor* weight,
                                  const aclTensor* runningMean, const aclTensor* runningVar, const aclTensor* saveMean,
                                  const aclTensor* saveInvstd, bool training, float eps, aclTensor** gradInput,
                                  aclTensor** gradWeight, aclTensor** gradBias, aclOpExecutor* executor)
{
    aclnnStatus bnPrepareResp = FastBatchNormBackwardPrepare(input, weight, runningMean, runningVar, saveMean,
                                                             saveInvstd, executor);
    CHECK_RET(bnPrepareResp == ACLNN_SUCCESS, bnPrepareResp);

    size_t dimNum = input->GetViewShape().GetDimNum();
    // fast batch norm: prepare input tensors for processing
    auto gradOutPre = gradOut;
    auto inputPre = input;
    if (dimNum < BN2D_INPUT_DIMS) {
        gradOutPre = op::ResizeFromND(gradOut, executor);
        CHECK_RET(gradOutPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        inputPre = op::ResizeFromND(input, executor);
        CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // fast batch norm: process backward computation
    aclTensor* result = nullptr;
    if (isBatchNormSupportAscendC()) {
        auto bnResult = FastBatchNormBackwardProcForAscendC(gradOutPre, inputPre, weight, runningMean, runningVar,
                                                            saveMean, saveInvstd, training, eps, &result, gradWeight,
                                                            gradBias, executor);
        CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
    } else {
        auto bnResult = FastBatchNormBackwardProc(gradOutPre, inputPre, weight, runningMean, runningVar, saveMean,
                                                  saveInvstd, training, eps, &result, gradWeight, gradBias, executor);
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

aclnnStatus FastBatchNormBackwardWeightAndBiasPost(const aclBoolArray* outputMask, aclTensor* bnGradWeight,
                                                   aclTensor* gradWeight, aclTensor* bnGradBias, aclTensor* gradBias,
                                                   aclOpExecutor* executor)
{
    if (!isBatchNormSupportAscendC()) {
        if ((*outputMask)[GRAD_WEIGHT_INDEX]) {
            auto viewCopyWeight = op::ResizeTo1D(bnGradWeight, gradWeight, isBatchNormSupportNcdhw(), executor);
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

// 空tensor早退发生在参数校验之前，访问outputMask与输出指针前需自行判空防越界；
// 空batch时gradWeight/gradBias数学上恒为0，需清零输出，避免调用方读到未初始化显存（issue#7080）。
static aclnnStatus FillGradWeightGradBiasWithZero(const aclBoolArray* outputMask, aclTensor* gradWeight,
                                                  aclTensor* gradBias, aclOpExecutor* executor)
{
    if (outputMask == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (outputMask->Size() > GRAD_WEIGHT_INDEX && (*outputMask)[GRAD_WEIGHT_INDEX] && gradWeight != nullptr &&
        !gradWeight->IsEmpty()) {
        auto ret = op::ProcessEmptyTensorWithValue(gradWeight, 0, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    if (outputMask->Size() > GRAD_BIAS_INDEX && (*outputMask)[GRAD_BIAS_INDEX] && gradBias != nullptr &&
        !gradBias->IsEmpty()) {
        auto ret = op::ProcessEmptyTensorWithValue(gradBias, 0, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* input,
                                                       const aclTensor* weight, const aclTensor* runningMean,
                                                       const aclTensor* runningVar, const aclTensor* saveMean,
                                                       const aclTensor* saveInvstd, bool training, double eps,
                                                       const aclBoolArray* outputMask, int version,
                                                       aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFastBatchNormBackward,
                   DFX_IN(gradOut, input, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps,
                          outputMask, version),
                   DFX_OUT(gradInput, gradWeight, gradBias));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckInputNotNull(gradOut, input), ACLNN_ERR_PARAM_NULLPTR);

    if (input->IsEmpty() || gradOut->IsEmpty()) {
        auto fillRet = FillGradWeightGradBiasWithZero(outputMask, gradWeight, gradBias, uniqueExecutor.get());
        CHECK_RET(fillRet == ACLNN_SUCCESS, fillRet);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = CheckParams(gradOut, input, gradInput, gradWeight, gradBias, outputMask, weight, runningMean, runningVar,
                           saveMean, saveInvstd, training);
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
    } else if ((inputShape.GetDimNum() == BN2D_INPUT_DIMS) && (input->GetStorageFormat() == Format::FORMAT_ND)) {
        // 4D ND与NCHW布局等价，ReFormat为物理格式，避免ND流入内置tiling段错误（issue#7080）
        inputContiguous = l0op::ReFormat(inputContiguous, Format::FORMAT_NCHW);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutContigous = l0op::ReFormat(gradOutContigous, Format::FORMAT_NCHW);
        CHECK_RET(gradOutContigous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if ((inputShape.GetDimNum() == MAX_BN_DIMS) && (input->GetStorageFormat() == Format::FORMAT_ND)) {
        // 5D ND与NCDHW布局等价，ReFormat为物理格式统一走NCDHW路径
        inputContiguous = l0op::ReFormat(inputContiguous, Format::FORMAT_NCDHW);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        gradOutContigous = l0op::ReFormat(gradOutContigous, Format::FORMAT_NCDHW);
        CHECK_RET(gradOutContigous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    aclTensor* bnGradInput = nullptr;
    aclTensor* bnGradWeight = nullptr;
    aclTensor* bnGradBias = nullptr;
    aclnnStatus bnResult = FastBatchNormBackward(gradOutContigous, inputContiguous, weight, runningMean, runningVar,
                                                 saveMean, saveInvstd, training, eps, &bnGradInput, &bnGradWeight,
                                                 &bnGradBias, uniqueExecutor.get());
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);
    auto weightResult = FastBatchNormBackwardWeightAndBiasPost(outputMask, bnGradWeight, gradWeight, bnGradBias,
                                                               gradBias, uniqueExecutor.get());
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

aclnnStatus aclnnFastBatchNormBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFastBatchNormBackward);
    // 固定写法，调用框架能力，完成计算
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
