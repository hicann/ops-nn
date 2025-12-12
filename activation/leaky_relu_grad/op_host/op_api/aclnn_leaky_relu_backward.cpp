/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_kernels/cast.h"
#include "leaky_relu_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_leaky_relu_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_DIM = 8;
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static const std::initializer_list<DataType> ASCEND910_SCALAR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL,   DataType::DT_INT16};

static const std::initializer_list<DataType> ASCEND910B_SCALAR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_BOOL,   DataType::DT_INT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope, const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(negativeSlope, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static const std::initializer_list<DataType>& GetScalarSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910B_SCALAR_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_SCALAR_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope)
{
    // 检查输入的数据类型是否在LeakyReluGrad算子的支持列表内
    auto supportList = GetDtypeSupportList();
    auto scalarList = GetScalarSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(negativeSlope, scalarList, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, broadcastShape, return false);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckIsResult(const aclScalar* negativeSlope, bool selfIsResult)
{
    if (selfIsResult && negativeSlope->ToDouble() < 0.0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "In-place leakyRelu backward calculation is triggered with a negativeSlope which is not supported, "
            "while negativeSlope: [%e], selfIsResult: [%d]. This is caused by calling in-place forward function "
            "with a negative slope, please call out-of-place version instead.",
            negativeSlope->ToDouble(), selfIsResult);
        return false;
    }
    return true;
}

static inline bool CheckPromoteType(const DataType gradOutputDtype, const DataType selfDtype, const DataType outDtype)
{
    // 检查self和other能否做数据类型推导
    DataType promoteType = PromoteType(gradOutputDtype, selfDtype);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected gradOutputDtype %s and selfDtype %s can promote dtype but check fail.",
            ToString(gradOutputDtype).GetString(), ToString(selfDtype).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope, bool selfIsResult,
    const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, negativeSlope, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOutput, self, negativeSlope), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor的shape是否异常，输出和输入的shape是否相同
    CHECK_RET(CheckShape(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查selfIsResult和negativeSlope取值
    CHECK_RET(CheckIsResult(negativeSlope, selfIsResult), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查gradOutput和self能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(
        CheckPromoteType(gradOutput->GetDataType(), self->GetDataType(), out->GetDataType()), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope, bool selfIsResult,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLeakyReluBackward, DFX_IN(gradOutput, self, negativeSlope, selfIsResult), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(gradOutput, self, negativeSlope, selfIsResult, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入为空tensor时，输出也是空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    DataType promoteType = PromoteType(gradOutput->GetDataType(), self->GetDataType());

    // 将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将gradOutput转换成隐式数据类型
    auto gradOutputCast = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入tensor转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将self转换成隐式数据类型
    auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用LeakyReluGrad算子kernel，完成计算
    auto output = l0op::LeakyReluGrad(gradOutputCast, selfCast, negativeSlope->ToFloat(), uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(output, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto copyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeakyReluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLeakyReluBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
