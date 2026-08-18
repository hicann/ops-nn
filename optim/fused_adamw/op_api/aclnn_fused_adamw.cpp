/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fused_adamw.cpp
 * \brief
 */

#include "aclnn_fused_adamw.h"
#include "aclnn_kernels/contiguous.h"
#include "fused_adamw.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> STEP_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                            op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensorList* paramsRef, const aclTensorList* grads, const aclTensorList* expAvgsRef,
                         const aclTensorList* expAvgSqsRef, const aclTensorList* stateSteps,
                         const aclTensorList* maxExpAvgSqsRef)
{
    OP_CHECK_NULL(paramsRef, return false);
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        OP_CHECK_NULL((*paramsRef)[i], return false);
    }
    OP_CHECK_NULL(grads, return false);
    for (uint64_t i = 0; i < grads->Size(); i++) {
        OP_CHECK_NULL((*grads)[i], return false);
    }
    OP_CHECK_NULL(expAvgsRef, return false);
    for (uint64_t i = 0; i < expAvgsRef->Size(); i++) {
        OP_CHECK_NULL((*expAvgsRef)[i], return false);
    }
    OP_CHECK_NULL(expAvgSqsRef, return false);
    for (uint64_t i = 0; i < expAvgSqsRef->Size(); i++) {
        OP_CHECK_NULL((*expAvgSqsRef)[i], return false);
    }
    OP_CHECK_NULL(stateSteps, return false);
    for (uint64_t i = 0; i < stateSteps->Size(); i++) {
        OP_CHECK_NULL((*stateSteps)[i], return false);
    }
    if (maxExpAvgSqsRef != nullptr) {
        for (uint64_t i = 0; i < maxExpAvgSqsRef->Size(); i++) {
            OP_CHECK_NULL((*maxExpAvgSqsRef)[i], return false);
        }
    }
    return true;
}

static bool CheckTensorListCount(const aclTensorList* paramsRef, const aclTensorList* grads,
                                 const aclTensorList* expAvgsRef, const aclTensorList* expAvgSqsRef,
                                 const aclTensorList* stateSteps, const aclTensorList* maxExpAvgSqsRef)
{
    auto tensorCount = paramsRef->Size();
    if (tensorCount == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paramsRef tensor list is empty.");
        return false;
    }
    if (grads->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grads tensor count does not match params.");
        return false;
    }
    if (expAvgsRef->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expAvgs tensor count does not match params.");
        return false;
    }
    if (expAvgSqsRef->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expAvgSqs tensor count does not match params.");
        return false;
    }
    if (stateSteps->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stateSteps tensor count does not match params.");
        return false;
    }
    if (maxExpAvgSqsRef != nullptr && maxExpAvgSqsRef->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxExpAvgSqs tensor count does not match params.");
        return false;
    }
    return true;
}

static bool CheckDtype(const aclTensorList* paramsRef, const aclTensorList* grads, const aclTensorList* expAvgsRef,
                       const aclTensorList* expAvgSqsRef, const aclTensorList* stateSteps,
                       const aclTensorList* maxExpAvgSqsRef)
{
    auto paramsTensor = (*paramsRef)[0];
    auto stateStepsTensor = (*stateSteps)[0];
    OP_CHECK_DTYPE_NOT_SUPPORT(paramsTensor, INPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(stateStepsTensor, STEP_DTYPE_SUPPORT_LIST, return false);
    op::DataType stepType = stateStepsTensor->GetDataType();
    for (uint64_t i = 1; i < stateSteps->Size(); i++) {
        if ((*stateSteps)[i]->GetDataType() != stepType) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all stateSteps tensors with the same dtype.");
            return false;
        }
    }
    op::DataType inputType = paramsTensor->GetDataType();
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        if ((*paramsRef)[i]->GetDataType() != inputType || (*grads)[i]->GetDataType() != inputType ||
            (*expAvgsRef)[i]->GetDataType() != inputType || (*expAvgSqsRef)[i]->GetDataType() != inputType) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same dtype.");
            return false;
        }
    }
    if (maxExpAvgSqsRef != nullptr) {
        auto maxExpAvgSqsTensor = (*maxExpAvgSqsRef)[0];
        OP_CHECK_DTYPE_NOT_SUPPORT(maxExpAvgSqsTensor, INPUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(paramsTensor, maxExpAvgSqsTensor, return false);
        for (uint64_t i = 0; i < maxExpAvgSqsRef->Size(); i++) {
            if ((*maxExpAvgSqsRef)[i]->GetDataType() != inputType) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same dtype.");
                return false;
            }
        }
    }
    return true;
}

static void CheckOptionalTensorEmpty(const aclTensor*& tensor)
{
    if (tensor == nullptr) {
        OP_LOGI("gradScaleOptional is nullptr");
        return;
    }
    if (tensor->IsEmpty()) {
        OP_LOGI("gradScaleOptional is empty, treating as nullptr.");
        tensor = nullptr;
    }
}

static bool CheckAttr(double lr, double beta1, double beta2, double weightDecay, double eps, bool amsgrad,
                      const aclTensorList* maxExpAvgSqsRef)
{
    if (lr < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The lr[%f] shoule be greater or equal than 0", lr);
        return false;
    }
    if (beta1 < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The beta1[%f] shoule be greater or equal than 0", beta1);
        return false;
    }
    if (beta2 < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The beta2[%f] shoule be greater or equal than 0", beta2);
        return false;
    }
    if (weightDecay < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The weightDecay[%f] shoule be greater or equal than 0", weightDecay);
        return false;
    }
    const float EPS = 1e-7f;
    if (eps > EPS || eps < 0) {
        OP_LOGW("An incorrect value for eps[%f] will affect accuracy, a value of 1e-8 is recommended.", eps);
    }
    if (amsgrad == true && maxExpAvgSqsRef == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When amsgrad is true, maxExpAvgSqsRef should not be nullptr.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensorList* paramsRef, const aclTensorList* grads, const aclTensorList* expAvgsRef,
                       const aclTensorList* expAvgSqsRef, const aclTensorList* maxExpAvgSqsRef)
{
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        op::Shape expectShape = (*paramsRef)[i]->GetViewShape();
        if ((*grads)[i]->GetViewShape() != expectShape || (*expAvgsRef)[i]->GetViewShape() != expectShape ||
            (*expAvgSqsRef)[i]->GetViewShape() != expectShape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same shape.");
            return false;
        }
        if (maxExpAvgSqsRef != nullptr && (*maxExpAvgSqsRef)[i]->GetViewShape() != expectShape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same shape.");
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* paramsRef, const aclTensorList* grads,
                               const aclTensorList* expAvgsRef, const aclTensorList* expAvgSqsRef,
                               const aclTensorList* maxExpAvgSqsRef, const aclTensorList* stateSteps, double lr,
                               double beta1, double beta2, double weightDecay, double eps, bool amsgrad)
{
    CHECK_RET(CheckAttr(lr, beta1, beta2, weightDecay, eps, amsgrad, maxExpAvgSqsRef), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotNull(paramsRef, grads, expAvgsRef, expAvgSqsRef, stateSteps, maxExpAvgSqsRef),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListCount(paramsRef, grads, expAvgsRef, expAvgSqsRef, stateSteps, maxExpAvgSqsRef),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(paramsRef, grads, expAvgsRef, expAvgSqsRef, stateSteps, maxExpAvgSqsRef),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(paramsRef, grads, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* FlattenDimsForTensor(const aclTensor* tensor, aclOpExecutor* executor)
{
    op::Shape shapeTensor = tensor->GetViewShape();
    int64_t dimNum = shapeTensor.GetDimNum();

    op::Shape newShape;
    int64_t catdimSize = 1;
    for (int64_t i = 0; i < dimNum; i++) {
        catdimSize *= shapeTensor.GetDim(i);
    }
    newShape.AppendDim(catdimSize);
    auto reshapeTensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    reshapeTensor->SetViewShape(newShape);
    reshapeTensor->SetOriginalShape(newShape);
    reshapeTensor->SetStorageShape(newShape);
    return reshapeTensor;
}

static const aclTensorList* MakeContiguousTensorList(const aclTensorList* tensorList, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> contiguousTensors;
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i]->IsEmpty()) {
            continue;
        }
        auto contiguous = l0op::Contiguous((*tensorList)[i], executor);
        CHECK_RET(contiguous != nullptr, nullptr);
        contiguous = FlattenDimsForTensor(contiguous, executor);
        contiguousTensors.emplace_back(contiguous);
    }
    return executor->AllocTensorList(contiguousTensors.data(), contiguousTensors.size());
}

static const aclTensorList* CastStateStepsToFloat(const aclTensorList* stateSteps, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> castedTensors;
    for (uint64_t i = 0; i < stateSteps->Size(); i++) {
        auto casted = l0op::Cast((*stateSteps)[i], DataType::DT_FLOAT, executor);
        CHECK_RET(casted != nullptr, nullptr);
        castedTensors.emplace_back(casted);
    }
    return executor->AllocTensorList(castedTensors.data(), castedTensors.size());
}

static void ViewCopyTensorList(const aclTensorList* src, const aclTensorList* dst, aclOpExecutor* executor)
{
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < dst->Size(); i++) {
        if ((*dst)[i]->IsEmpty()) {
            continue;
        }
        l0op::ViewCopy((*src)[cnt], (*dst)[i], executor);
        cnt += 1;
    }
}
} // namespace

aclnnStatus aclnnFusedAdamwGetWorkspaceSize(const aclTensorList* paramsRef, const aclTensorList* grads,
                                            const aclTensorList* expAvgsRef, const aclTensorList* expAvgSqsRef,
                                            const aclTensorList* maxExpAvgSqsRef, const aclTensorList* stateSteps,
                                            const aclTensor* gradScaleOptional, const aclTensor* foundInfOptional,
                                            double lr, double beta1, double beta2, double weightDecay, double eps,
                                            bool amsgrad, bool maximize, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedAdamw,
                   DFX_IN(paramsRef, grads, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef, stateSteps, gradScaleOptional,
                          foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
                   DFX_OUT(paramsRef, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(paramsRef, grads, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef, stateSteps, lr, beta1, beta2,
                           weightDecay, eps, amsgrad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CheckOptionalTensorEmpty(gradScaleOptional);
    CheckOptionalTensorEmpty(foundInfOptional);

    if (gradScaleOptional != nullptr) {
        gradScaleOptional = l0op::Cast(gradScaleOptional, DataType::DT_FLOAT, uniqueExecutor.get());
    }

    auto paramsContiguous = MakeContiguousTensorList(paramsRef, uniqueExecutor.get());
    CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradsContiguous = MakeContiguousTensorList(grads, uniqueExecutor.get());
    CHECK_RET(gradsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto expAvgsRefContiguous = MakeContiguousTensorList(expAvgsRef, uniqueExecutor.get());
    CHECK_RET(expAvgsRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto expAvgSqsRefContiguous = MakeContiguousTensorList(expAvgSqsRef, uniqueExecutor.get());
    CHECK_RET(expAvgSqsRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensorList* maxExpAvgSqsRefContiguous = nullptr;
    if (maxExpAvgSqsRef != nullptr) {
        maxExpAvgSqsRefContiguous = MakeContiguousTensorList(maxExpAvgSqsRef, uniqueExecutor.get());
        CHECK_RET(maxExpAvgSqsRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensorList* stateStepsFloat = CastStateStepsToFloat(stateSteps, uniqueExecutor.get());
    CHECK_RET(stateStepsFloat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto [paramsOut, expAvgsOut, expAvgSqsOut, maxExpAvgSqsOut] = l0op::FusedAdamw(
        paramsContiguous, gradsContiguous, expAvgsRefContiguous, expAvgSqsRefContiguous, maxExpAvgSqsRefContiguous,
        stateStepsFloat, gradScaleOptional, foundInfOptional, static_cast<float>(lr), static_cast<float>(beta1),
        static_cast<float>(beta2), static_cast<float>(weightDecay), static_cast<float>(eps), amsgrad, maximize,
        uniqueExecutor.get());
    CHECK_RET(paramsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(expAvgsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(expAvgSqsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ViewCopyTensorList(paramsOut, paramsRef, uniqueExecutor.get());
    ViewCopyTensorList(expAvgsOut, expAvgsRef, uniqueExecutor.get());
    ViewCopyTensorList(expAvgSqsOut, expAvgSqsRef, uniqueExecutor.get());
    if (maxExpAvgSqsRef != nullptr) {
        ViewCopyTensorList(maxExpAvgSqsOut, maxExpAvgSqsRef, uniqueExecutor.get());
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedAdamw(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedAdamw);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
