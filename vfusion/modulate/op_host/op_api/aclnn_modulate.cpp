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
 * \file aclnn_modulate.cpp
 * \brief
 */

#include "aclnn_modulate.h"
#include "modulate.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static bool CheckNotNull(const aclTensor* self, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    }
    return NULL_SUPPORT_LIST;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* scaleOptional, const aclTensor* shiftOptional, aclTensor* out)
{
    // 检查self，scaleOptional，shiftOptional, out的数据类型是否在算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    // self，scaleOptional和shiftOptional的数据类型要一致
    if (scaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scaleOptional, supportList, return false);
        if (self->GetDataType() != scaleOptional->GetDataType()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dtype of self [%s], scaleOptional [%s] need to be the same.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(scaleOptional->GetDataType()).GetString());
            return false;
        }
    }
    if (shiftOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(shiftOptional, supportList, return false);
        if (self->GetDataType() != shiftOptional->GetDataType()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dtype of self [%s], shiftOptional [%s] need to be the same.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(shiftOptional->GetDataType()).GetString());
            return false;
        }
    }
    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The dtype of self [%s], out [%s] need to be the same.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* scaleOptional, const aclTensor* shiftOptional, aclTensor* out)
{
    op::Shape selfShape = self->GetViewShape();
    size_t selfDimNum = selfShape.GetDimNum();
    // 对非ND数据格式增加warning
    if (self->GetStorageFormat() != op::Format::FORMAT_ND || scaleOptional->GetStorageFormat() != op::Format::FORMAT_ND 
                                                    || shiftOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("aclnnModulate only support ND format");
    }
    // self的维度必须为3
    if (selfDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of self must be 3, but got %zu.", selfDimNum);
        return false;
    }
    // 拦截self为空，scaleOptional或shiftOptional不为空的情况
    if (self->IsEmpty() && (!scaleOptional->IsEmpty() || !shiftOptional->IsEmpty())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when self is empty, scaleOptional and shiftOptional must be empty");
        return false;
    }
    auto CheckOptionalTensor = [selfShape](const aclTensor* optionalTensor, const char* tensorName) -> bool {
        if (optionalTensor == nullptr) {
            return true;
        }
        op::Shape optionalShape = optionalTensor->GetViewShape();
        size_t optionalDimNum = optionalShape.GetDimNum();
        // 可选tensor的维度是否为2或3
        if (optionalDimNum != 2 && optionalDimNum != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of %s and shiftOptional must be 2 or 3, but got %zu.",
                    tensorName, optionalDimNum);
            return false;
        }
        // 第一和最后一个维度是否匹配
        if (optionalDimNum == 2) {
            if ((selfShape.GetDim(0) != optionalShape.GetDim(0)) || (selfShape.GetDim(2) != optionalShape.GetDim(1))) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first and last dimension of self and %s must be the same.", tensorName);
                return false;
            }
        } else {
            if ((selfShape.GetDim(0) != optionalShape.GetDim(0)) || (selfShape.GetDim(2) != optionalShape.GetDim(2))) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first and last dimension of self and %s must be the same.", tensorName);
                return false;
            }
            // 可选参数维度为3时，第二维必须为1
            if (optionalShape.GetDim(1) != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when the dimension of %s is 3, the second dimension must be 1.", tensorName);
                return false;
            }
        }
        return true;
    };
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, self->GetViewShape(), return false);
    return CheckOptionalTensor(scaleOptional, "scaleOptional") && CheckOptionalTensor(shiftOptional, "shiftOptional");
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* scaleOptional, const aclTensor* shiftOptional, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, scaleOptional, shiftOptional, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的shape是否合法
    CHECK_RET(CheckShape(self, scaleOptional, shiftOptional, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetOptTensorContiguous(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnModulateGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scaleOptional, const aclTensor* shiftOptional, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnModulate, DFX_IN(self, scaleOptional, shiftOptional), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, scaleOptional, shiftOptional, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    bool hasEmptyTensor = self->IsEmpty() || (scaleOptional != nullptr && scaleOptional->IsEmpty()) ||
                          (shiftOptional != nullptr && shiftOptional->IsEmpty());
    if (hasEmptyTensor) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 若可选参数都未给出，直接返回self
    if (scaleOptional == nullptr && shiftOptional == nullptr) {
        // 将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入scaleOptional转换成连续的tensor
    auto scaleOptionalContiguous = GetOptTensorContiguous(scaleOptional, uniqueExecutor.get());
    // 将输入shiftOptional转换成连续的tensor
    auto shiftOptionalContiguous = GetOptTensorContiguous(shiftOptional, uniqueExecutor.get());

    // 执行L0算子
    auto ModulateRes =
        l0op::Modulate(selfContiguous, scaleOptionalContiguous, shiftOptionalContiguous, uniqueExecutor.get());
    CHECK_RET(ModulateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出data上
    auto viewCopyResult = l0op::ViewCopy(ModulateRes, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModulate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnModulate);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif