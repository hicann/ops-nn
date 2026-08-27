/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_copy.h"
#include "foreach_copy.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_FLOAT16, DataType::DT_BF16,   DataType::DT_INT32,
    DataType::DT_INT8,   DataType::DT_UINT8,   DataType::DT_INT16,  DataType::DT_UINT16,
    DataType::DT_UINT32, DataType::DT_INT64,   DataType::DT_DOUBLE, DataType::DT_BOOL};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static inline bool CheckNotNull(const aclTensorList* x, const aclTensorList* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* x, const aclTensorList* out)
{
    for (uint64_t i = 0; i < x->Size(); i++) {
        if (IsPrivateFormat((*x)[i]->GetStorageFormat()) || IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return EMPTY_LIST;
    }
}

static inline bool IsValidDtypeMapping(DataType xDtype, DataType outDtype)
{
    if (xDtype == outDtype) {
        return true;
    }
    if (xDtype == DataType::DT_FLOAT && (outDtype == DataType::DT_FLOAT16 || outDtype == DataType::DT_BF16)) {
        return true;
    }
    if (xDtype == DataType::DT_FLOAT16 && outDtype == DataType::DT_FLOAT) {
        return true;
    }
    if (xDtype == DataType::DT_BF16 && outDtype == DataType::DT_FLOAT) {
        return true;
    }
    return false;
}

static inline bool CheckDtypeValid(const aclTensorList* x, const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }

    if (x->Size() == 0 || out->Size() == 0) {
        return true;
    }

    auto xDtype = (*x)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*x)[0], dtypeSupportList, return false);

    for (uint64_t i = 0; i < x->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x)[i], xDtype, return false);
    }

    auto outDtype = (*out)[0]->GetDataType();
    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], outDtype, return false);
    }

    if (!IsValidDtypeMapping(xDtype, outDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dtype mapping from x(%d) to out(%d) is not supported",
                static_cast<int32_t>(xDtype), static_cast<int32_t>(outDtype));
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* x, const aclTensorList* out)
{
    if (x->Size() != out->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors");
        return false;
    }

    for (uint64_t i = 0; i < x->Size(); i++) {
        OP_CHECK_MAX_DIM((*x)[i], MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MAX_DIM((*out)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* x, const aclTensorList* out)
{
    CHECK_RET(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachCopyGetWorkspaceSize(const aclTensorList* x, const aclTensorList* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto contiguousTensors = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ForeachCopy(contiguousTensors, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachCopyGetWorkspaceSize(const aclTensorList* x, aclTensorList* out, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachCopy, DFX_IN(x), DFX_OUT(out));
    return ExecForeachCopyGetWorkspaceSize(x, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachCopy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
