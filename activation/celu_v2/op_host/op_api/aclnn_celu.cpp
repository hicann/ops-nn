/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_celu.h"
#include "celu_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
}

static inline bool CheckNotNull(const aclTensor* self, const aclScalar* alpha, const aclTensor* out)
{
    // self、alpha、out不能为空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclScalar* alpha, const aclTensor* out)
{
    // 检查数据类型是否在CeluV2算子的支持列表内
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查alpha的数据类型能否转换为FLOAT
    if (!CanCast(alpha->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "alpha dtype %s can not cast to float32.",
            ToString(alpha->GetDataType()).GetString());
        return false;
    }

    // 检查self的数据类型能否转换为out的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    return true;
}

static inline bool CheckAttributeValue(const aclScalar* alpha)
{
    // 当alpha为0时，报错
    if (!(alpha->ToFloat() > 0 || alpha->ToFloat() < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha cannot be 0 for CELU.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* alpha, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, alpha, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查alpha的取值是否满足要求
    CHECK_RET(CheckAttributeValue(alpha), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果self是空tensor，则out也是空tensor，直接返回
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的Tensor
    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用CeluV2算子kernel
    auto celuV2Out = l0op::CeluV2(contiguousSelf, alpha->ToFloat(), uniqueExecutor.get());
    CHECK_RET(celuV2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(celuV2Out, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCeluGetWorkspaceSize(
    const aclTensor* self, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnCelu, DFX_IN(self, alpha), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, alpha, out, workspaceSize, executor);
}

aclnnStatus aclnnCelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCelu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceCeluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceCelu, DFX_IN(selfRef, alpha), DFX_OUT(selfRef));
    return GetWorkspaceSizeCommon(selfRef, alpha, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceCelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceCelu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
