/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_sgd.h"
#include "aclnn_kernels/contiguous.h"
#include "fused_sgd.h"
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

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListFromSocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201: {
            return ASCEND910B_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_3510: {
            return ASCEND950_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool CheckNotNull(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef)
{
    OP_CHECK_NULL(paramsRef, return false);
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        OP_CHECK_NULL((*paramsRef)[i], return false);
    }
    OP_CHECK_NULL(gradsRef, return false);
    for (uint64_t i = 0; i < gradsRef->Size(); i++) {
        OP_CHECK_NULL((*gradsRef)[i], return false);
    }
    if(momentumBufferListOptionalRef != nullptr) {
        for (uint64_t i = 0; i < momentumBufferListOptionalRef->Size(); i++) {
            OP_CHECK_NULL((*momentumBufferListOptionalRef)[i], return false);
        }
    }
    return true;
}

static bool CheckTensorListCount(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef)
{
    auto tensorCount = paramsRef->Size();
    if (tensorCount == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paramsRef tensor list is empty.");
        return false;
    }
    if (gradsRef->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradsRef tensor count does not match paramsRef.");
        return false;
    }
    if (momentumBufferListOptionalRef != nullptr && momentumBufferListOptionalRef->Size() != tensorCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "momentumBufferListOptionalRef tensor count does not match paramsRef.");
        return false;
    }
    return true;
}

static bool CheckDtype(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef)
{
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListFromSocVersion();
    auto paramsTensor = (*paramsRef)[0];

    OP_CHECK_DTYPE_NOT_SUPPORT(paramsTensor, dtypeSupportList, return false);
    op::DataType inputType = paramsTensor->GetDataType();
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        if ((*paramsRef)[i]->GetDataType() != inputType || (*gradsRef)[i]->GetDataType() != inputType) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same dtype.");
            return false;
        }
    }
    if(momentumBufferListOptionalRef != nullptr) {
        auto momentumTensor = (*momentumBufferListOptionalRef)[0];
        OP_CHECK_DTYPE_NOT_SUPPORT(momentumTensor, dtypeSupportList, return false);
        OP_CHECK_DTYPE_NOT_SAME(paramsTensor, momentumTensor, return false);
        for (uint64_t i = 0; i < momentumBufferListOptionalRef->Size(); i++) {
            if ((*momentumBufferListOptionalRef)[i]->GetDataType() != inputType) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same dtype.");
                return false;
            }
        }
    }
    return true;
}

static void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList)
{
    if (tensorList == nullptr) {
        OP_LOGI("momentumBufferListOptionalRef is nullptr");
        return;
    }
    if (tensorList->Size() == 0) {
        OP_LOGI("momentumBufferListOptionalRef is set nullptr because len(momentumBufferListOptionalRef) is 0.");
        tensorList = nullptr;
    }
}

static void CheckOptionalTensorEmpty(const aclTensor *&tensor)
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

static void CheckIsFirstStep(bool isFirstStep)
{
    if (isFirstStep) {
        OP_LOGW("isFirstStep argument has no effect when momentumBufferListOptionalRef is empty");
    }
}

static bool CheckAttr(
    const aclTensorList *momentumBufferListOptionalRef, 
    float weightDecay, float momentum, float lr, 
    float dampening, bool nesterov) 
{
    if (weightDecay < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weightDecay[%f] shoule be greater or equal than 0", weightDecay);
        return false;
    }
    if (momentum < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "momentum[%f] shoule be greater or equal than 0", momentum);
        return false;
    }
    if (lr < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lr[%f] shoule be greater or equal than 0", lr);
        return false;
    }
    const float EPS = 1e-6f;
    if (nesterov && (momentum <= 0 || std::abs(dampening) >= EPS)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "nesterov[%d] momentum requires a momentum[%f] and zero dampening[%f].", static_cast<int32_t>(nesterov), momentum, dampening);
        return false;
    }
    if ((momentumBufferListOptionalRef == nullptr && std::abs(momentum) >= EPS)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "momentum[%f] is invalid. momentum shoube be 0 when momentumBufferListOptionalRef is nullptr.", momentum);
        return false;
    }
    if ((momentumBufferListOptionalRef != nullptr && (momentum < 0.0 || std::abs(momentum) < EPS))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "momentum[%f] is invalid. momentum shoube be greater than 0 when momentumBufferListOptionalRef is not nullptr.", momentum);
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef) 
{
    for (uint64_t i = 0; i < paramsRef->Size(); i++) {
        op::Shape expectShape = (*paramsRef)[i]->GetViewShape();
        if((*gradsRef)[i]->GetViewShape() != expectShape || 
            (momentumBufferListOptionalRef != nullptr && 
            (*momentumBufferListOptionalRef)[i]->GetViewShape() != expectShape)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same shape.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef,
    const aclTensor *gradScaleOptional,
    float weightDecay, float momentum, float lr, float dampening, bool nesterov)
{
    CHECK_RET(CheckAttr(momentumBufferListOptionalRef, weightDecay, momentum, lr, dampening, nesterov), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotNull(paramsRef, gradsRef, momentumBufferListOptionalRef), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListCount(paramsRef, gradsRef, momentumBufferListOptionalRef), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(paramsRef, gradsRef, momentumBufferListOptionalRef), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(paramsRef, gradsRef, momentumBufferListOptionalRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* FlattenDims(const aclTensor* tensor, aclOpExecutor* executor)
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

static const aclTensorList* MakeContiguousTensorList(
    const aclTensorList* tensorList, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> contiguousTensors;
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i]->IsEmpty()) {
            continue;
        }
        auto contiguous = l0op::Contiguous((*tensorList)[i], executor);
        CHECK_RET(contiguous != nullptr, nullptr);
        contiguous = FlattenDims(contiguous, executor);
        contiguousTensors.emplace_back(contiguous);
    }
    return executor->AllocTensorList(contiguousTensors.data(), contiguousTensors.size());
}

static void ViewCopyTensorList(
    const aclTensorList* src, const aclTensorList* dst, aclOpExecutor* executor)
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

aclnnStatus aclnnFusedSgdGetWorkspaceSize(
    const aclTensorList *paramsRef,
    const aclTensorList *gradsRef,
    const aclTensorList *momentumBufferListOptionalRef,
    const aclTensor *gradScaleOptional,
    float weightDecay,
    float momentum,
    float lr,
    float dampening,
    bool nesterov,
    bool maximize,
    bool isFirstStep,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedSgd,
        DFX_IN(paramsRef, gradsRef, momentumBufferListOptionalRef, gradScaleOptional,
               weightDecay, momentum, lr, dampening, nesterov, maximize, isFirstStep),
        DFX_OUT(paramsRef, gradsRef, momentumBufferListOptionalRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (paramsRef->Size() == 0) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CheckOptionalTensorListEmpty(momentumBufferListOptionalRef);
    CheckOptionalTensorEmpty(gradScaleOptional);

    CheckIsFirstStep(isFirstStep);

    auto ret = CheckParams(paramsRef, gradsRef, momentumBufferListOptionalRef, gradScaleOptional, weightDecay, momentum, lr, dampening, nesterov);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradScaleOptional != nullptr) {
        gradScaleOptional = l0op::Cast(gradScaleOptional, DataType::DT_FLOAT, uniqueExecutor.get());
    }

    auto paramsContiguous = MakeContiguousTensorList(paramsRef, uniqueExecutor.get());
    CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradsContiguous = MakeContiguousTensorList(gradsRef, uniqueExecutor.get());
    CHECK_RET(gradsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensorList* momentumContiguous = nullptr;
    if(momentumBufferListOptionalRef != nullptr) {
        momentumContiguous = MakeContiguousTensorList(momentumBufferListOptionalRef, uniqueExecutor.get());
    }

    auto [paramsOut, gradsOut, momentumOut] = l0op::FusedSgd(
        paramsContiguous, gradsContiguous, momentumContiguous,
        gradScaleOptional,
        weightDecay, momentum, lr, dampening, nesterov, maximize, isFirstStep,
        uniqueExecutor.get());
    CHECK_RET(paramsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gradsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(momentumContiguous == nullptr || momentumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ViewCopyTensorList(paramsOut, paramsRef, uniqueExecutor.get());
    ViewCopyTensorList(gradsOut, gradsRef, uniqueExecutor.get());
    if(momentumBufferListOptionalRef != nullptr) {
        ViewCopyTensorList(momentumOut, momentumBufferListOptionalRef, uniqueExecutor.get());
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedSgd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedSgd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
