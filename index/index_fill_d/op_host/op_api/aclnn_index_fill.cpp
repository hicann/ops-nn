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
 * \file aclnn_index_fill.cpp
 * \brief
 */

#include "aclnn_index_fill.h"
#include "index_fill.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/framework_op.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<op::DataType> DTYPE_A2_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_INDEX_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor* self, const aclTensor* index, const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(
    const aclTensor* self, int64_t dim, const aclTensor* out)
{
    if (self->IsEmpty()) {
        return true;
    }
    // 校验self的shape是否等于out的shape
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    // 最大维度限制
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);

    auto selfShape = self->GetViewShape();
    auto selfDim = static_cast<int64_t>(selfShape.GetDimNum());
    if ((dim != 0 && dim >= selfDim) || (dim == 0 && dim > selfDim)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim[%ld] is greater than self dim[%ld].", dim, selfDim);
        return false;
    }

    if ((dim < 0 && (dim * (-1)) > selfDim && selfDim > 0) || (dim < 0 && (dim * (-1)) > 1 && selfDim == 0)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dim value error, abs(input dim[%ld]) is greater than self dim[%ld].", dim,
            selfDim);
        return false;
    }
    return true;
}

static bool CheckIndexDtypeValid(const aclTensor* index)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        OP_CHECK_DTYPE_NOT_SUPPORT(index, DTYPE_INDEX_SUPPORT_LIST, return true);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_A2_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, NULL_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, index, value, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckIndexDtypeValid(index), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的shape是否满足要求
    CHECK_RET(CheckShape(self, dim, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool NeedTranspose(const aclTensor* self, int64_t dim)
{
    if (self->GetDataType() == op::DataType::DT_INT64 || self->GetDataType() == op::DataType::DT_BOOL) {
        auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
        dim = dim > 0 ? dim : dim + selfDimNum;
        if (dim == selfDimNum - 1) {
            return true;
        }
    }
    return false;
}

static const aclTensor* SwapDim(const aclTensor* self, aclOpExecutor* executor)
{
    // 交换self的首轴与尾轴
    auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    std::vector<int64_t> perm(selfDimNum);
    for (int64_t i = 0; i < selfDimNum; ++i) {
        perm[i] = i;
    }
    std::swap(perm[0], perm[selfDimNum - 1]);
    auto valuePerm = executor->AllocIntArray(perm.data(), selfDimNum);
    OP_CHECK_NULL(valuePerm, return nullptr);
    auto selfTrans = l0op::Transpose(self, valuePerm, executor);
    return selfTrans;
}

aclnnStatus aclnnIndexFillGetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnIndexFill, DFX_IN(self, dim, index, value), DFX_OUT(out));
    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, index, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (index->Size() == 0) {
        auto viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
    CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 0维Tensor，转1维
    int64_t squeezeDim = 0;
    bool needUnsqueeze = (selfContiguous->GetViewShape().GetDimNum() == 0);
    selfContiguous =
        needUnsqueeze ? l0op::UnsqueezeNd(selfContiguous, squeezeDim, uniqueExecutor.get()) : selfContiguous;
    CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "squeeze failed!");
    needUnsqueeze = (indexContiguous->GetViewShape().GetDimNum() == 0);
    indexContiguous =
        needUnsqueeze ? l0op::UnsqueezeNd(indexContiguous, squeezeDim, uniqueExecutor.get()) : indexContiguous;
    CHECK_COND(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "squeeze failed!");
    // 将value 转为和self同类型的Tensor
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(value, selfContiguous->GetDataType());
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // self为bool和int64时，暂时只支持非尾轴计算，遇到尾轴需转置首轴
    bool needTranspose = NeedTranspose(selfContiguous, dim);
    if (needTranspose) {
        selfContiguous = SwapDim(selfContiguous, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dim = 0;
    }

    // 进行IndexFill计算
    auto indexFillOut = l0op::IndexFill(selfContiguous, indexContiguous, valueTensor, dim, uniqueExecutor.get());
    CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (needTranspose) {
        indexFillOut = SwapDim(indexFillOut, uniqueExecutor.get());
        CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(indexFillOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexFill(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexFill);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(
    aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnIndexFillGetWorkspaceSize(selfRef, dim, index, value, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceIndexFill(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceIndexFill);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif