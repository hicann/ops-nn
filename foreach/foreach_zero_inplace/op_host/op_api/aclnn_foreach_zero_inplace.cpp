/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_zero_inplace.h"
#include <initializer_list>
#include "foreach_zero_inplace_l0op.h"
#include "foreach_contiguous_helper.h"
#include "foreach_null_check_helper.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr uint64_t MAX_TENSOR_COUNT = 256;
static const std::initializer_list<DataType> TENSOR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT16, DataType::DT_INT32};
static const std::initializer_list<DataType> KIRIN_TENSOR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT16, DataType::DT_INT32};
static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND950:
            return TENSOR_DTYPE_SUPPORT_LIST;
        case SocVersion::KIRINX90:
        case SocVersion::KIRIN9030:
            return KIRIN_TENSOR_DTYPE_SUPPORT_LIST;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ForeachZeroInplace is not supported on the current platform.");
            return EMPTY_LIST;
    }
}

static aclnnStatus CheckParams(const aclTensorList* x, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListNotNull(x, "x"), ACLNN_ERR_PARAM_INVALID);

    const auto& dtypeSupportList = GetDtypeSupportList();
    CHECK_RET(dtypeSupportList.size() != 0, ACLNN_ERR_PARAM_INVALID);
    if (x->Size() > MAX_TENSOR_COUNT) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "The tensor list x must contain at most 256 tensors.");
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    if (x->Size() == 0) {
        return ACLNN_SUCCESS;
    }

    auto dtype = (*x)[0]->GetDataType();
    for (uint64_t i = 0; i < x->Size(); i++) {
        auto tensor = (*x)[i];
        OP_CHECK_DTYPE_NOT_SUPPORT(tensor, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(tensor, dtype, return ACLNN_ERR_INNER_TILING_ERROR);
        OP_CHECK_MAX_DIM(tensor, MAX_SUPPORT_DIMS_NUMS, return ACLNN_ERR_INNER_TILING_ERROR);
        if (IsPrivateFormat(tensor->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private formats are not supported for x.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static bool AllViewsEmpty(const aclTensorList* x)
{
    for (uint64_t i = 0; i < x->Size(); i++) {
        if ((*x)[i]->GetViewShape().GetShapeSize() != 0) {
            return false;
        }
    }
    return true;
}

static aclnnStatus ExecForeachZeroInplaceGetWorkspaceSize(const aclTensorList* x, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(x, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (AllViewsEmpty(x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto contiguousTensors = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ret = l0op::ForeachZeroInplace(contiguousTensors, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousTensors, x, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(const aclTensorList* x, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachZeroInplace, DFX_IN(x), DFX_OUT(x));
    return ExecForeachZeroInplaceGetWorkspaceSize(x, workspaceSize, executor);
}

aclnnStatus aclnnForeachZeroInplace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachZeroInplace);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
