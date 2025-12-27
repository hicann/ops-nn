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
 * \file aclnn_sigmoid.cpp
 * \brief
 */

#include "aclnn_sigmoid.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "sigmoid.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_INT16,      op::DataType::DT_INT32,      op::DataType::DT_INT64,  op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64,  op::DataType::DT_COMPLEX128, op::DataType::DT_BF16,   op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> DTYPE_OUT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64,  op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_DOUBLE,    DataType::DT_COMPLEX64,
    DataType::DT_COMPLEX128};

static const std::initializer_list<DataType>& GetSelfRefDtypeList() {
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return DTYPE_OUT_LIST;
    } else {
        return ASCEND910_OUTPUT_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckInplaceDtypeValid(aclTensor *selfRef) {
    auto inplaceSupportList = GetSelfRefDtypeList();
    // 检查selfRef的数据类型是否在inplace sigmoid算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, inplaceSupportList, return false);

    return true;
}

inline static bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

inline static bool CheckNotNull(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 检查self的数据类型是否在sigmoid算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_OUT_LIST, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

inline static bool CheckShape(const aclTensor *self, const aclTensor *out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    // self的维度必须小于 9
    OP_CHECK_MAX_DIM(self, 8, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(aclTensor *selfRef) {
    OP_CHECK_NULL(selfRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 检查selfRef的数据类型是否在inplace sigmoid算子的支持列表内
    CHECK_RET(CheckInplaceDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSigmoidGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                               aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // sigmoid算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用cast算子将不支持的类型转化为float
    auto castDtype = selfContiguous->GetDataType();
    if (!CheckType(castDtype, DTYPE_OUT_LIST)) {
        castDtype = out->GetDataType();
    }
    auto selfCast = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用Sigmoid算子Kernel
    auto sigmoidOpOut = l0op::Sigmoid(selfCast, uniqueExecutor.get());
    CHECK_RET(sigmoidOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(sigmoidOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSigmoid, DFX_IN(self), DFX_OUT(out));
    return ExecSigmoidGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSigmoidGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize,
                                                aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInplaceSigmoid, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto ret = CheckInplaceParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto out = const_cast<aclTensor*>(selfRef);
    return ExecSigmoidGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSigmoid);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceSigmoid);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif