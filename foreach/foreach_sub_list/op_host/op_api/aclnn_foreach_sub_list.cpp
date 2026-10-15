/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_sub_list.h"
#include "foreach_sub_list_v2.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<ge::DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8};

// alpha(aclTensor*)的dtype须与x1的dtype按def映射一致（DtypeScalarToTensor2：FLOAT16→FLOAT16，BFLOAT16→FLOAT，整型→INT32）
static const std::initializer_list<ge::DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {ge::DT_FLOAT};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {ge::DT_FLOAT16};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {ge::DT_INT32};

static const std::initializer_list<ge::DataType> EMPTY_LIST = {};

static inline bool CheckNotNull(const aclTensorList* self, const aclTensorList* x2, const aclTensor* alpha,
                                const aclTensorList* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* self, const aclTensorList* x2, const aclTensorList* out)
{
    for (uint64_t i = 0; i < self->Size(); i++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*self)[i]->GetStorageFormat()) || IsPrivateFormat((*x2)[i]->GetStorageFormat()) ||
            IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<ge::DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return EMPTY_LIST;
    }
}

static inline bool CheckDtypeValid(const aclTensorList* self, const aclTensorList* x2, const aclTensor* alpha,
                                   const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }
    if (self->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    auto selfDtyte = (*self)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*self)[0], dtypeSupportList, return false);

    // check the releation of self and alpha
    if (selfDtyte == ge::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(alpha, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
    } else if (selfDtyte == ge::DT_FLOAT || selfDtyte == ge::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(alpha, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(alpha, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }

    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*self)[i], selfDtyte, return false);
    }

    for (uint64_t i = 0; i < x2->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], selfDtyte, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtyte, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* self, const aclTensorList* x2, const aclTensorList* out)
{
    // tensorlist size检查
    if (self->Size() != x2->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors, got %lu and %lu",
                self->Size(), x2->Size());
        return false;
    }

    // tensorlist中tensor shape一致性检查
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*self)[i], (*x2)[i], return false);
    }

    // tensor 维度检查
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_MAX_DIM((*self)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*self)[i], (*out)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* self, const aclTensorList* x2, const aclTensor* alpha,
                                      const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, x2, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtypeValid/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(self, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, x2, alpha, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, x2, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(self, x2, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachSubListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                      const aclTensor* alpha, const aclTensorList* out,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x1->Size() == 0 || x2->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensorsX1 = ForeachMakeContiguousTensorList(x1, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto contiguousTensorsX2 = ForeachMakeContiguousTensorList(x2, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 复用V2桥接，alpha(aclTensor*)直接透传，输出到连续buffer
    auto result = l0op::ForeachSubListV2(contiguousTensorsX1, contiguousTensorsX2, alpha, contiguousOut,
                                         uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachSubListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                const aclTensor* alpha, aclTensorList* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachSubList, DFX_IN(x1, x2, alpha), DFX_OUT(out));
    return ExecForeachSubListGetWorkspaceSize(x1, x2, alpha, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachSubList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachSubList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
