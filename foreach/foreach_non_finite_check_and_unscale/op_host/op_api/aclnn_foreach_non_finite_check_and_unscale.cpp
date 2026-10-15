/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_non_finite_check_and_unscale.h"
#include <initializer_list>
#include "foreach_non_finite_check_and_unscale.h"
#include "foreach_contiguous_helper.h"
#include "foreach_null_check_helper.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr uint64_t MAX_TENSOR_COUNT = 256;
static const std::initializer_list<DataType> TENSOR_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                          DataType::DT_BF16};
static const std::initializer_list<DataType> REDUCED_TENSOR_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                  DataType::DT_FLOAT16};
static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND950:
            return TENSOR_DTYPE_SUPPORT_LIST;
        case SocVersion::ASCEND310P:
        case SocVersion::KIRINX90:
        case SocVersion::KIRIN9030:
            return REDUCED_TENSOR_DTYPE_SUPPORT_LIST;
        default:
            break;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ForeachNonFiniteCheckAndUnscale is not supported on the current platform.");
    return EMPTY_LIST;
}

static aclnnStatus CheckTensorShapeAndFormat(const aclTensor* tensor, const char* name)
{
    OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return ACLNN_ERR_INNER_TILING_ERROR);
    if (IsPrivateFormat(tensor->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private formats are not supported for %s.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensorList* scaledGrads, const aclTensor* foundInf, const aclTensor* invScale,
                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_NULL(scaledGrads, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(foundInf, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(invScale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListNotNull(scaledGrads, "scaledGrads"), ACLNN_ERR_PARAM_INVALID);

    const auto& dtypeSupportList = GetDtypeSupportList();
    CHECK_RET(dtypeSupportList.size() != 0, ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(foundInf, DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(invScale, DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    if (scaledGrads->Size() == 0 || scaledGrads->Size() > MAX_TENSOR_COUNT) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "scaledGrads must contain between 1 and 256 tensors.");
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    auto dtype = (*scaledGrads)[0]->GetDataType();
    for (uint64_t i = 0; i < scaledGrads->Size(); i++) {
        auto tensor = (*scaledGrads)[i];
        OP_CHECK_DTYPE_NOT_SUPPORT(tensor, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(tensor, dtype, return ACLNN_ERR_INNER_TILING_ERROR);
        if (tensor->GetViewShape().GetShapeSize() == 0) {
            OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "scaledGrads must not contain empty tensors.");
            return ACLNN_ERR_INNER_TILING_ERROR;
        }
        auto ret = CheckTensorShapeAndFormat(tensor, "scaledGrads");
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    if (foundInf->GetViewShape().GetShapeSize() != 1 || invScale->GetViewShape().GetShapeSize() != 1) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "foundInf and invScale must each contain exactly one element.");
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    auto ret = CheckTensorShapeAndFormat(foundInf, "foundInf");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckTensorShapeAndFormat(invScale, "invScale");
}

static aclnnStatus ExecForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList* scaledGrads,
                                                                       const aclTensor* foundInf,
                                                                       const aclTensor* invScale,
                                                                       uint64_t* workspaceSize,
                                                                       aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(scaledGrads, foundInf, invScale, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto exec = uniqueExecutor.get();
    auto contiguousGrads = ForeachMakeContiguousTensorList(scaledGrads, exec);
    CHECK_RET(contiguousGrads != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Single-element views may share larger storage and still require normalization.
    const aclTensor* foundItems[] = {foundInf};
    const aclTensor* scaleItems[] = {invScale};
    auto foundList = exec->AllocTensorList(foundItems, 1);
    auto scaleList = exec->AllocTensorList(scaleItems, 1);
    CHECK_RET(foundList != nullptr && scaleList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Preserve the current foundInf value; the kernel leaves it untouched for finite gradients.
    auto contiguousFoundList = ForeachMakeContiguousTensorList(foundList, exec);
    auto contiguousScaleList = ForeachMakeContiguousTensorList(scaleList, exec);
    CHECK_RET(contiguousFoundList != nullptr && contiguousScaleList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = l0op::ForeachNonFiniteCheckAndUnscale(contiguousGrads, (*contiguousFoundList)[0], (*contiguousScaleList)[0],
                                                exec);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousGrads, scaledGrads, exec), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousFoundList, foundList, exec), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList* scaledGrads,
                                                                 const aclTensor* foundInf, const aclTensor* invScale,
                                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachNonFiniteCheckAndUnscale, DFX_IN(scaledGrads, foundInf, invScale),
                   DFX_OUT(scaledGrads, foundInf));
    return ExecForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(scaledGrads, foundInf, invScale, workspaceSize,
                                                               executor);
}

aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachNonFiniteCheckAndUnscale);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
