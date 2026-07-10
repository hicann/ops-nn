/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_addcdiv_scalar.h"
#include "foreach_addcdiv_scalar_v2.h"
#include "../../foreach_utils/op_api/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType> SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_DOUBLE};
static const std::initializer_list<DataType> SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2201 && !Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return EMPTY_LIST;
    }
    return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
}

static inline bool CheckNotNull(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                const aclTensor* scalar, const aclTensorList* out)
{
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x3, return false);
    OP_CHECK_NULL(scalar, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                               const aclTensorList* out)
{
    for (uint64_t i = 0; i < x1->Size(); i++) {
        if (IsPrivateFormat((*out)[i]->GetStorageFormat()) || IsPrivateFormat((*x1)[i]->GetStorageFormat()) ||
            IsPrivateFormat((*x2)[i]->GetStorageFormat()) || IsPrivateFormat((*x3)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                   const aclTensor* scalar, const aclTensorList* out)
{
    if (x1->Size() == 0) {
        return true;
    }

    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }

    auto selfDtype = (*x1)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*x1)[0], dtypeSupportList, return false);

    if (selfDtype == DataType::DT_BF16 || selfDtype == DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else if (selfDtype == DataType::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, SCALAR_FLOAT16_SUPPORT_LIST, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtype, return false);
    }
    for (uint64_t i = 0; i < x3->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x3)[i], selfDtype, return false);
    }
    for (uint64_t i = 0; i < x2->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], selfDtype, return false);
    }
    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x1)[i], selfDtype, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                              const aclTensorList* out)
{
    if (x1->Size() != x2->Size() || x1->Size() != x3->Size() || x1->Size() != out->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors");
        return false;
    }

    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*x1)[i], (*out)[i], return false);
    }

    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_MAX_DIM((*x1)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*x1)[i], (*x2)[i], return false);
        OP_CHECK_SHAPE_NOT_EQUAL((*x1)[i], (*x3)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                      const aclTensor* scalar, const aclTensorList* out)
{
    CHECK_RET(CheckNotNull(x1, x2, x3, scalar, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x1, x2, x3, scalar, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x1, x2, x3, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x1, x2, x3, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachAddcdivScalarGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                            const aclTensorList* x3, const aclTensor* scalar,
                                                            const aclTensorList* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, x3, scalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x1->Size() == 0 || x2->Size() == 0 || x3->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensorsX3 = ForeachMakeContiguousTensorList(x3, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousTensorsX2 = ForeachMakeContiguousTensorList(x2, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousTensorsX1 = ForeachMakeContiguousTensorList(x1, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 复用V2桥接，scalar(aclTensor*)直接透传，输出到连续buffer
    auto result = l0op::ForeachAddcdivScalarV2(contiguousTensorsX1, contiguousTensorsX2, contiguousTensorsX3, scalar,
                                               contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachAddcdivScalarGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                      const aclTensorList* x3, const aclTensor* scalar,
                                                      aclTensorList* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachAddcdivScalar, DFX_IN(x1, x2, x3, scalar), DFX_OUT(out));
    return ExecForeachAddcdivScalarGetWorkspaceSize(x1, x2, x3, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachAddcdivScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachAddcdivScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
