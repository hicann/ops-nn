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
 * \file aclnn_fused_linear_cross_entropy_loss_grad.cpp
 * \brief
 */
#include "aclnn_fused_linear_cross_entropy_loss_grad.h"
#include "fused_linear_cross_entropy_loss_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

#define OP_CHECK_FORMAT_NOT_ND(tensor, retExpr) \
  if ((tensor)->GetStorageFormat() != Format::FORMAT_ND) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s only supports ND format.", #tensor); \
    retExpr; \
  }

static constexpr size_t EXPECTED_DIM_ONE = 1;
static constexpr size_t EXPECTED_DIM_TWO = 2;
static constexpr size_t DIM_NUM_ZERO = 0;
static constexpr size_t DIM_NUM_ONE = 1;
static constexpr size_t SIZE_8 = 8;
static constexpr size_t SIZE_128 = 128;

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> HALF_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};
static const std::initializer_list<op::DataType> BOOL_DTYPE_SUPPORT_LIST = {DataType::DT_BOOL, DataType::DT_UINT8};

namespace {
struct FusedLinearCrossEntropyLossGradInputs {
    const aclTensor* grad;
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* targetMask;
    const aclTensor* maskedTarget;
    float labelSmoothing;
    const aclTensor* logitsMaxOptional;
    const aclTensor* sumExpLogitsOptional;
    const aclTensor* softmaxOptional;
};

struct FusedLinearCrossEntropyLossGradOutputs {
    aclTensor* inputGradOut;
    aclTensor* weightGradOut;
};
}

inline static bool CheckNotNull(FusedLinearCrossEntropyLossGradInputs& inputs,
                                FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    if (inputs.softmaxOptional == nullptr and \
        (inputs.logitsMaxOptional == nullptr or inputs.sumExpLogitsOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "softmaxOptional can not be nullptr when logitsMaxOptional or sumExpLogitsOptional is nullptr");
        return false;
    }
    OP_CHECK_NULL(inputs.grad, return false);
    OP_CHECK_NULL(inputs.input, return false);
    OP_CHECK_NULL(inputs.weight, return false);
    OP_CHECK_NULL(inputs.targetMask, return false);
    OP_CHECK_NULL(inputs.maskedTarget, return false);

    OP_CHECK_NULL(outputs.inputGradOut, return false);
    OP_CHECK_NULL(outputs.weightGradOut, return false);
    return true;
}


inline static bool CheckDtypeValid(FusedLinearCrossEntropyLossGradInputs& inputs,
                                   FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.grad, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.input, HALF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.weight, HALF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.targetMask, BOOL_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputs.maskedTarget, INT_DTYPE_SUPPORT_LIST, return false);
    if (inputs.softmaxOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputs.softmaxOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputs.logitsMaxOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(inputs.sumExpLogitsOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SAME(inputs.input, inputs.weight, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputs.input, outputs.inputGradOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputs.input, outputs.weightGradOut, return false);
    return true;
}

inline static bool CheckFormatValid(FusedLinearCrossEntropyLossGradInputs& inputs,
                                    FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    OP_CHECK_FORMAT_NOT_ND(inputs.grad, return false);
    OP_CHECK_FORMAT_NOT_ND(inputs.input, return false);
    OP_CHECK_FORMAT_NOT_ND(inputs.weight, return false);
    OP_CHECK_FORMAT_NOT_ND(inputs.targetMask, return false);
    OP_CHECK_FORMAT_NOT_ND(inputs.maskedTarget, return false);
    if (inputs.softmaxOptional) {
        OP_CHECK_FORMAT_NOT_ND(inputs.softmaxOptional, return false);
    } else {
        OP_CHECK_FORMAT_NOT_ND(inputs.logitsMaxOptional, return false);
        OP_CHECK_FORMAT_NOT_ND(inputs.sumExpLogitsOptional, return false);
    }
    OP_CHECK_FORMAT_NOT_ND(outputs.inputGradOut, return false);
    OP_CHECK_FORMAT_NOT_ND(outputs.weightGradOut, return false);
    return true;
}

inline static bool CheckAttrValid(FusedLinearCrossEntropyLossGradInputs& inputs,
                                  FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    if (inputs.labelSmoothing != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnFusedLinearCrossEntropyLossGrad currently only support labelSmoothing equals 0.");
        return false;
    }
    return true;
}

inline static bool CheckShapeValid(FusedLinearCrossEntropyLossGradInputs& inputs,
                                   FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    // 维度数量检查
    OP_CHECK_WRONG_DIMENSION(inputs.grad, EXPECTED_DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(inputs.input, EXPECTED_DIM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(inputs.weight, EXPECTED_DIM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(inputs.targetMask, EXPECTED_DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(inputs.maskedTarget, EXPECTED_DIM_ONE, return false);
    if (inputs.softmaxOptional) {
        OP_CHECK_WRONG_DIMENSION(inputs.softmaxOptional, EXPECTED_DIM_TWO, return false);
    } else {
        OP_CHECK_WRONG_DIMENSION(inputs.logitsMaxOptional, EXPECTED_DIM_ONE, return false);
        OP_CHECK_WRONG_DIMENSION(inputs.sumExpLogitsOptional, EXPECTED_DIM_ONE, return false);
    }
    OP_CHECK_WRONG_DIMENSION(outputs.inputGradOut, EXPECTED_DIM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(outputs.weightGradOut, EXPECTED_DIM_TWO, return false);

    // 维度长度检查
    op::Shape gradShape = inputs.grad->GetViewShape();
    op::Shape inputShape = inputs.input->GetViewShape();
    op::Shape weightShape = inputs.weight->GetViewShape();
    op::Shape targetMaskShape = inputs.targetMask->GetViewShape();
    op::Shape maskedTargetShape = inputs.maskedTarget->GetViewShape();
    op::Shape inputGradOutShape = outputs.inputGradOut->GetViewShape();
    op::Shape weightGradOutShape = outputs.weightGradOut->GetViewShape();
    auto BT = gradShape.GetDim(DIM_NUM_ZERO);
    auto H = inputShape.GetDim(DIM_NUM_ONE);
    auto V = weightShape.GetDim(DIM_NUM_ZERO);

    if (inputShape.GetDim(DIM_NUM_ZERO) != BT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input.size(0) should be equal to grad.size(0)");
        return false;
    }
    if (weightShape.GetDim(DIM_NUM_ZERO) < SIZE_128 and inputs.softmaxOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight.size(0) < 128 is not supported");
        return false;
    }
    if (weightShape.GetDim(DIM_NUM_ONE) != H) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight.size(1) should be equal to input.size(1)");
        return false;
    }
    if (inputs.targetMask->GetDataType() == DataType::DT_BOOL) {
        if (targetMaskShape.GetDim(DIM_NUM_ZERO) != BT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetMask.size(0) should be equal to grad.size(0) when targetMask is in BOOL");
            return false;
        }
    } else {
        if (targetMaskShape.GetDim(DIM_NUM_ZERO) * SIZE_8 < BT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetMask.size(0) * 8 should not be less than grad.size(0) when targetMask is in UINT8");
            return false;
        }
    }
    if (maskedTargetShape.GetDim(DIM_NUM_ZERO) != BT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskedTarget.size(0) should be equal to grad.size(0)");
        return false;
    }
    op::Shape shapeBTV = {BT, V};
    op::Shape shapeBTH = {BT, H};
    op::Shape shapeVH = {V, H};
    op::Shape shapeBT = {BT};
    if (inputs.softmaxOptional) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputs.softmaxOptional, shapeBTV, return false);
    } else {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputs.logitsMaxOptional, shapeBT, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputs.sumExpLogitsOptional, shapeBT, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputs.inputGradOut, shapeBTH, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputs.weightGradOut, shapeVH, return false);
    return true;
}

inline static bool CheckPlatformSupported()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or \
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return true;
    }
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Current platform is not supported");
    return false;
}

inline static aclnnStatus CheckParam(FusedLinearCrossEntropyLossGradInputs& inputs,
                                     FusedLinearCrossEntropyLossGradOutputs& outputs)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(inputs, outputs), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查Dtype是否支持
    CHECK_RET(CheckDtypeValid(inputs, outputs), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查Format是否支持
    CHECK_RET(CheckFormatValid(inputs, outputs), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查属性取值是否支持
    CHECK_RET(CheckAttrValid(inputs, outputs), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(inputs, outputs), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static bool CheckTupleNotNullptr(std::tuple<aclTensor*, aclTensor*>& result)
{
    if (std::get<DIM_NUM_ZERO>(result) == nullptr or \
        std::get<DIM_NUM_ONE>(result) == nullptr) {
        return false;
    }
    return true;
}

static const aclTensor* ViewCopyTensors(
    FusedLinearCrossEntropyLossGradOutputs& outputs,
    std::tuple<aclTensor*, aclTensor*>& result,
    aclOpExecutor* executor)
{
    auto viewCopyResult = l0op::ViewCopy(std::get<DIM_NUM_ZERO>(result),
                                        outputs.inputGradOut, executor);
    CHECK_RET(viewCopyResult != nullptr, nullptr);
    viewCopyResult = l0op::ViewCopy(std::get<DIM_NUM_ONE>(result),
                                    outputs.weightGradOut, executor);
    CHECK_RET(viewCopyResult != nullptr, nullptr);
    return viewCopyResult;
}

aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor *grad, const aclTensor *input, const aclTensor *weight, const aclTensor *targetMask, const aclTensor *maskedTarget,
    float labelSmoothing, const aclTensor *logitsMaxOptional, const aclTensor *sumExpLogitsOptional, const aclTensor *softmaxOptional,
    aclTensor *inputGradOut, aclTensor *weightGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFusedLinearCrossEntropyLossGrad,
        DFX_IN(grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional),
        DFX_OUT(inputGradOut, weightGradOut)
    );
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedLinearCrossEntropyLossGradInputs inputs = {
        grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional};
    FusedLinearCrossEntropyLossGradOutputs outputs = {inputGradOut, weightGradOut};

    // 入参检查
    auto ret = CheckParam(inputs, outputs);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (input->IsEmpty() || weight->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 转连续
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    auto targetMaskContiguous = l0op::Contiguous(targetMask, uniqueExecutor.get());
    auto maskedTargetContiguous = l0op::Contiguous(maskedTarget, uniqueExecutor.get());
    const aclTensor *logitsMaxOptionalContiguous = nullptr;;
    const aclTensor *sumExpLogitsOptionalContiguous = nullptr;
    const aclTensor *softmaxOptionalContiguous = nullptr;;
    if (inputs.softmaxOptional) {
        // 高性能分支
        softmaxOptionalContiguous = l0op::Contiguous(softmaxOptional, uniqueExecutor.get());
    } else {
        // 省显存分支
        logitsMaxOptionalContiguous = l0op::Contiguous(logitsMaxOptional, uniqueExecutor.get());
        sumExpLogitsOptionalContiguous = l0op::Contiguous(sumExpLogitsOptional, uniqueExecutor.get());
    }

    // 6. 检查平台是否支持(调整检验时机增加覆盖率)
    CHECK_RET(CheckPlatformSupported(), ACLNN_ERR_RUNTIME_ERROR);

    // 调用l0算子
    std::tuple<aclTensor *, aclTensor *> result = l0op::FusedLinearCrossEntropyLossGrad(
        gradContiguous, inputContiguous, weightContiguous,
        targetMaskContiguous, maskedTargetContiguous, logitsMaxOptionalContiguous, sumExpLogitsOptionalContiguous,
        softmaxOptionalContiguous, labelSmoothing,
        uniqueExecutor.get()
    );
    CHECK_RET(CheckTupleNotNullptr(result), ACLNN_ERR_INNER_NULLPTR);

    // 获取输出视图
    auto viewCopyResult = ViewCopyTensors(outputs, result, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedLinearCrossEntropyLossGrad);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
