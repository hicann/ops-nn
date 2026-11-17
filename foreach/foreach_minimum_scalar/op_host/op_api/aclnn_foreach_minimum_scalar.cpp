/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_minimum_scalar.h"
#include "foreach_minimum_scalar_v2.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "opdev/platform.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include "op_api/op_api_def_nn.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32};

// scalar(aclTensor*)的dtype须与x的dtype按def映射一致（DtypeScalarToTensor2：FLOAT16→FLOAT16，BFLOAT16→FLOAT，整型→INT32）
static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16};

static const std::initializer_list<DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {DataType::DT_INT32};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static inline bool CheckNull(const aclTensorList* minimumScalarSelf, const aclTensor* scalar, const aclTensorList* out)
{
    OP_CHECK_NULL(minimumScalarSelf, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckTensorListSize(const aclTensorList* minimumScalarSelf, const aclTensorList* out)
{
    // self 和 out 的 tensor 个数必须一致，避免后续按下标访问 out 越界
    if (minimumScalarSelf->Size() != out->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors, got %lu and %lu",
                minimumScalarSelf->Size(), out->Size());
        return false;
    }
    return true;
}

static inline bool CheckFormat(const aclTensorList* minimumScalarSelf, const aclTensorList* out)
{
    for (uint64_t m = 0; m < minimumScalarSelf->Size(); m++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*minimumScalarSelf)[m]->GetStorageFormat()) ||
            IsPrivateFormat((*out)[m]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto minimumScalarArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (Ops::NN::AclnnUtil::IsRegbase(minimumScalarArch) || minimumScalarArch == NpuArch::DAV_2201) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return EMPTY_LIST;
    }
}

static inline bool CheckDtypeValid(const aclTensorList* minimumScalarSelf, const aclTensor* scalar,
                                   const aclTensorList* out)
{
    const auto& dtypeSupportList_2 = GetDtypeSupportList();
    if (dtypeSupportList_2.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }
    if (minimumScalarSelf->Size() == 0) {
        return true;
    }
    auto minimumScalarDtype = (*minimumScalarSelf)[0]->GetDataType();
    // checkself input dtype, and check the releation of input and out
    OP_CHECK_DTYPE_NOT_SUPPORT((*minimumScalarSelf)[0], dtypeSupportList_2, return false);
    for (uint64_t k = 0; k < minimumScalarSelf->Size(); k++) {
        OP_CHECK_DTYPE_NOT_MATCH((*minimumScalarSelf)[k], minimumScalarDtype, return false);
    }

    for (uint64_t k = 0; k < out->Size(); k++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[k], minimumScalarDtype, return false);
    }

    // check the releation of self and scalar
    if (minimumScalarDtype == DataType::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
    } else if (minimumScalarDtype == DataType::DT_FLOAT || minimumScalarDtype == DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckScalarShape(const aclTensor* scalar)
{
    // scalar 维度不超过8维，且只含一个元素（与 arch35 tiling 契约对齐）
    OP_CHECK_MAX_DIM(scalar, MAX_SUPPORT_DIMS_NUMS, return false);
    if (scalar->GetViewShape().GetShapeSize() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The scalar tensor must contain exactly one element, but got %ld.",
                scalar->GetViewShape().GetShapeSize());
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* minimumScalarSelf, const aclTensor* scalar, const aclTensorList* out)
{
    // tensor 维度检查
    for (uint64_t k = 0; k < minimumScalarSelf->Size(); k++) {
        OP_CHECK_MAX_DIM((*minimumScalarSelf)[k], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t k = 0; k < minimumScalarSelf->Size(); k++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*minimumScalarSelf)[k], (*out)[k], return false);
    }

    // scalar维度与元素个数检查
    if (!CheckScalarShape(scalar)) {
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* minimumScalarSelf, const aclTensor* scalar,
                                      const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNull(minimumScalarSelf, scalar, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtypeValid/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(minimumScalarSelf, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // self 和 out 的 tensor 个数必须一致（文档约束：Tensor 个数与 x 的 Tensor 个数相同）
    CHECK_RET(CheckTensorListSize(minimumScalarSelf, out), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(minimumScalarSelf, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(minimumScalarSelf, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(minimumScalarSelf, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachMinimumScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar,
                                                            const aclTensorList* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto minimumScalarExecutor = CREATE_EXECUTOR();
    CHECK_RET(minimumScalarExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret_3 = CheckParams(x, scalar, out);
    CHECK_RET(ret_3 == ACLNN_SUCCESS, ret_3);

    // 空Tensorlist处理
    if (x->Size() == 0) {
        *workspaceSize = 0;
        minimumScalarExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensors = ForeachMakeContiguousTensorList(x, minimumScalarExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, minimumScalarExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 复用V2桥接，scalar(aclTensor*)直接透传，输出到连续buffer
    auto result = l0op::ForeachMinimumScalarV2(contiguousTensors, scalar, contiguousOut, minimumScalarExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, minimumScalarExecutor.get()),
              ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = minimumScalarExecutor->GetWorkspaceSize();
    minimumScalarExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMinimumScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar,
                                                      aclTensorList* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachMinimumScalar, DFX_IN(x, scalar), DFX_OUT(out));
    return ExecForeachMinimumScalarGetWorkspaceSize(x, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachMinimumScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachMinimumScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
