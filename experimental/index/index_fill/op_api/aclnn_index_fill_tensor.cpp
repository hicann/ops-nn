/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_index_fill_tensor.cpp
 * \brief
 */

#include "aclnn_index_fill_tensor.h"
#include "index_fill.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/framework_op.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT, op::DataType::DT_INT64,
                                                                       op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> DTYPE_910B_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_DOUBLE};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* index, const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* index, int64_t dim, const aclTensor* out)
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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim[%ld] is greater than self dim[%ld].", dim,
                selfDim);
        return false;
    }

    if ((dim < 0 && selfDim > 0 && dim < -selfDim) || (dim < 0 && selfDim == 0 && dim < -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, abs(input dim[%ld]) is greater than self dim[%ld].", dim,
                selfDim);
        return false;
    }

    int64_t transferDim = dim >= 0 ? dim : (selfDim > 0 ? (dim + selfDim) : 0);
    for (int64_t i = 0; i < static_cast<int64_t>(index->Size()); i++) {
        auto dimSize = selfDim == 0 ? 1 : static_cast<int64_t>(selfShape.GetDim(transferDim));
        if ((*index)[i] >= dimSize || (*index)[i] < (-dimSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index value[%ld] is out of range, it should be smaller than [%ld].",
                    (*index)[i], dimSize);
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_910B_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

static bool CheckPromoteType(const aclScalar* value)
{
    if (IsComplexType(value->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Type of value do not support complex type");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t dim, const aclIntArray* index, const aclScalar* value,
                               const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, index, value, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的shape是否满足要求
    CHECK_RET(CheckShape(self, index, dim, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查value的类型
    CHECK_RET(CheckPromoteType(value), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self)
{
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnIndexFillTensor/aclnnInplaceIndexFillTensor doesn't support format NZ.");
    }
}

aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclIntArray* index,
                                                 const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnIndexFillTensor, DFX_IN(self, dim, index, value), DFX_OUT(out));
    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, index, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查格式
    CheckFormat(self);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 合并 Empty 和 index Size==0 场景。
    // 1. 显式初始化 *workspaceSize = 0 避免随机值引发 OOM。
    // 2. 统一执行 ViewCopy 保证 out 总是被正确刷新。
    if (self->IsEmpty() || index->Size() == 0) {
        *workspaceSize = 0;
        auto viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 负数dim转换
    auto selfDim = self->GetViewShape().GetDimNum();
    dim = dim >= 0 ? dim : (selfDim > 0 ? (dim + selfDim) : 0);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 1.将index从aclIntArray类型转为变成aclTensor类型
    auto indexTensor = uniqueExecutor.get()->ConvertToTensor(index, op::ToOpDataType(ACL_INT64));
    auto indexContiguous = l0op::Contiguous(indexTensor, uniqueExecutor.get());
    CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 2.将value转为和self同类型的Tensor
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(value, selfContiguous->GetDataType());

    // 3. 调用统一的 index_fill l0 接口 (全量走AICore实现)
    auto indexFillOut = l0op::IndexFill(selfContiguous, indexContiguous, valueTensor, dim, uniqueExecutor.get());
    CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(indexFillOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexFillTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexFillTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceIndexFillTensorGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclIntArray* index,
                                                        const aclScalar* value, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor)
{
    return aclnnIndexFillTensorGetWorkspaceSize(selfRef, dim, index, value, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceIndexFillTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceIndexFillTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
