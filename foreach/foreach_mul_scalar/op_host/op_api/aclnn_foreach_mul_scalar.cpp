/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aclnn_foreach_mul_scalar.h"
#include "foreach_mul_scalar_v2.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "op_api/op_api_def_nn.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> MUL_SCALAR_TENSOR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32};

// scalar(aclTensor*)的dtype须与x的dtype按def映射一致（DtypeScalarToTensor2：FLOAT16→FLOAT16，BFLOAT16→FLOAT，整型→INT32）
static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16};

static const std::initializer_list<DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {DataType::DT_INT32};

static const std::initializer_list<DataType> EMPTY_LIST = {};
static inline bool CheckNotNull(const aclTensorList* mulScalarSelf, const aclTensor* scalar, const aclTensorList* out)
{
    OP_CHECK_NULL(mulScalarSelf, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* mulScalarSelf, const aclTensorList* out)
{
    for (uint64_t j = 0; j < mulScalarSelf->Size(); j++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*mulScalarSelf)[j]->GetStorageFormat()) ||
            IsPrivateFormat((*out)[j]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase()) {
        return MUL_SCALAR_TENSOR_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for arch %u is not implemented",
                static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch()));
        return EMPTY_LIST;
    }
}

static inline bool CheckDtype(const aclTensorList* mulScalarSelf, const aclTensor* scalar, const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for arch %u is not implemented",
                static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch()));
        return false;
    }
    if (mulScalarSelf->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    OP_CHECK_NULL((*mulScalarSelf)[0], return false);
    auto mulScalarDtype = (*mulScalarSelf)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*mulScalarSelf)[0], dtypeSupportList, return false);
    for (uint64_t i = 0; i < mulScalarSelf->Size(); i++) {
        OP_CHECK_NULL((*mulScalarSelf)[i], return false);
        OP_CHECK_DTYPE_NOT_MATCH((*mulScalarSelf)[i], mulScalarDtype, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_NULL((*out)[i], return false);
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], mulScalarDtype, return false);
    }

    // check the releation of self and scalar
    if (mulScalarDtype == DataType::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
    } else if (mulScalarDtype == DataType::DT_FLOAT || mulScalarDtype == DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* mulScalarSelf, const aclTensorList* out)
{
    // tensorlist size检查
    OP_CHECK_TENSORLIST_SIZE_EQUAL(mulScalarSelf, out, return false);

    // tensor 维度检查
    for (uint64_t n = 0; n < mulScalarSelf->Size(); n++) {
        OP_CHECK_MAX_DIM((*mulScalarSelf)[n], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t n = 0; n < mulScalarSelf->Size(); n++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*mulScalarSelf)[n], (*out)[n], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* mulScalarSelf, const aclTensor* scalar,
                                      const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(mulScalarSelf, scalar, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtype/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(mulScalarSelf, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtype(mulScalarSelf, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(mulScalarSelf, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(mulScalarSelf, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachMulScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar,
                                                        const aclTensorList* out, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto mulScalarExecutor = CREATE_EXECUTOR();
    CHECK_RET(mulScalarExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret_1 = CheckParams(x, scalar, out);
    CHECK_RET(ret_1 == ACLNN_SUCCESS, ret_1);

    // 空Tensorlist处理
    if (x->Size() == 0) {
        *workspaceSize = 0;
        mulScalarExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensors = ForeachMakeContiguousTensorList(x, mulScalarExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, mulScalarExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 复用V2桥接，scalar(aclTensor*)直接透传，输出到连续buffer
    auto result = l0op::ForeachMulScalarV2(contiguousTensors, scalar, contiguousOut, mulScalarExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, mulScalarExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = mulScalarExecutor->GetWorkspaceSize();
    mulScalarExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMulScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar, aclTensorList* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachMulScalar, DFX_IN(x, scalar), DFX_OUT(out));
    return ExecForeachMulScalarGetWorkspaceSize(x, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachMulScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachMulScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
