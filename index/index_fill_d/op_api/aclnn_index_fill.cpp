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

#include <cstring>
#include "aclnn_index_fill.h"
#include "index_fill.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/inc/op_host/op_const_def_nn.h"
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
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;
constexpr int64_t TWO = 2;

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<op::DataType> DTYPE_910B_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_BF16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_950_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_960_SUPPORT_LIST = {
    op::DataType::DT_INT32,      op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,       op::DataType::DT_BF16,    op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_INT16,      op::DataType::DT_DOUBLE,  Ops::NN::DT_HIFLOAT4,   op::DataType::DT_FLOAT4_E2M1,
    op::DataType::DT_FLOAT4_E1M2};

// 960PR（DAV_9202）不支持 fp4 系列 dtype，且走 IndexFillD 兜底（tiling 仅支持基础 6 种 dtype）
static const std::initializer_list<op::DataType> DTYPE_960PR_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_INDEX_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                             op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor* self, const aclTensor* index, const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, int64_t dim, const aclTensor* out)
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

    if ((dim < 0 && (dim * (-1)) > selfDim && selfDim > 0) || (dim < 0 && (dim * (-1)) > 1 && selfDim == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, abs(input dim[%ld]) is greater than self dim[%ld].", dim,
                selfDim);
        return false;
    }
    return true;
}

static bool CheckIndexDtypeValid(const aclTensor* index)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(index, DTYPE_INDEX_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* value, const aclTensor* out)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_910B_SUPPORT_LIST, return false);
    } else if (npuArch == NpuArch::DAV_9201) {
        // 仅 960DT（DAV_9201）支持 fp4 系列 dtype
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_960_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(value, DTYPE_960_SUPPORT_LIST, return false);
    } else if (npuArch == NpuArch::DAV_9202) {
        // 960PR 不支持 fp4，dtype 能力与 IndexFillD 兜底内核对齐
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_960PR_SUPPORT_LIST, return false);
    } else if (Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_950_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, NULL_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

static bool CheckFp4Support(const aclTensor* self, const aclScalar* value, int64_t dim)
{
    if (self->GetDataType() != Ops::NN::DT_HIFLOAT4 && self->GetDataType() != op::DataType::DT_FLOAT4_E2M1 &&
        self->GetDataType() != op::DataType::DT_FLOAT4_E1M2) {
        return true;
    }

    if (self->GetDataType() != value->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The self and value must have the same data type when the dtype is DT_HIFLOAT4, DT_FLOAT4_E2M1, "
                "or DT_FLOAT4_E1M2.");
        return false;
    }

    auto selfShape = self->GetViewShape();
    auto selfDim = static_cast<int64_t>(selfShape.GetDimNum());
    int64_t actualDim = dim >= 0 ? dim : dim + selfDim;
    int64_t afterAxis = 1;
    for (int64_t i = actualDim + 1; i < selfDim; i++) {
        afterAxis *= selfShape.GetDim(i);
    }
    // hif4、fp4 时需满足input的dim轴之后的维度乘积为偶数。
    if (afterAxis % TWO != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The number of input in each line must be an even number when dtype of input is DT_HIFLOAT4, "
                "DT_FLOAT4_E2M1 and DT_FLOAT4_E1M2.");
        return false;
    }
    return true;
}

static bool IsNonContiguous(const aclTensor* tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    op::Shape shapeI = tensor->GetViewShape();
    auto dimNum = static_cast<int64_t>(shapeI.GetDimNum());
    // 0 维（scalar）张量恒视为连续，避免后续以 dimNum - 1 == -1 越界访问 stride
    if (dimNum <= 0) {
        return false;
    }
    auto shapeIStride = tensor->GetViewStrides();
    // 尾轴stride不为1，非连续
    if (shapeIStride[dimNum - 1] != 1) {
        return true;
    }
    // 其他轴的stride[i]!=shape[i+1]*stride[i+1]，非连续
    for (int64_t j = dimNum - TWO; j >= 0; j--) {
        if (shapeIStride[j] != shapeIStride[j + 1] * shapeI[j + 1]) {
            return true;
        }
    }

    return false;
}

static bool CheckContiguousSupport(const aclTensor* self)
{
    if (self->GetDataType() != Ops::NN::DT_HIFLOAT4 && self->GetDataType() != op::DataType::DT_FLOAT4_E2M1 &&
        self->GetDataType() != op::DataType::DT_FLOAT4_E1M2) {
        return true;
    }
    auto selfIsNonCon = IsNonContiguous(self);
    if (selfIsNonCon) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When self has dtype DT_HIFLOAT4, DT_FLOAT4_E2M1, or DT_FLOAT4_E1M2, only contiguous support is "
                "available.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t dim, const aclTensor* index, const aclScalar* value,
                               const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, index, value, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, value, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckIndexDtypeValid(index), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的shape是否满足要求
    CHECK_RET(CheckShape(self, dim, out), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查FP4支持情况
    CHECK_RET(CheckFp4Support(self, value, dim), ACLNN_ERR_PARAM_INVALID);
    // 6. 检查是否连续
    CHECK_RET(CheckContiguousSupport(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool NeedTranspose(const aclTensor* self, int64_t dim)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        // 950/960芯片不需要做转置操作
        return false;
    }

    auto dtype = self->GetDataType();
    if (dtype == op::DataType::DT_INT64 || dtype == op::DataType::DT_BOOL || dtype == op::DataType::DT_INT8 ||
        dtype == op::DataType::DT_UINT8 || dtype == op::DataType::DT_DOUBLE) {
        auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
        dim = dim >= 0 ? dim : dim + selfDimNum;
        if (dim == selfDimNum - 1) {
            return true;
        }
    }
    return false;
}

static bool NeedInt64View(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return self->GetDataType() == op::DataType::DT_DOUBLE &&
           (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93);
}

static aclTensor* CreateInt64View(const aclTensor* self, aclOpExecutor* executor)
{
    // Equal-width views preserve the bits and leave the caller's tensor metadata unchanged.
    auto view = executor->CreateView(self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(),
                                     self->GetViewOffset());
    OP_CHECK_NULL(view, return nullptr);
    view->SetDataType(op::DataType::DT_INT64);
    return view;
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

aclnnStatus ExecIndexFillGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                          const aclScalar* value, aclTensor* out, bool isInplace,
                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, index, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Reuse the complete INT64 path, including layout operations, without numerical casts.
    if (NeedInt64View(self)) {
        auto inputView = CreateInt64View(self, uniqueExecutor.get());
        CHECK_RET(inputView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto outputView = CreateInt64View(out, uniqueExecutor.get());
        CHECK_RET(outputView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const double doubleValue = value->ToDouble();
        int64_t valueBits = 0;
        std::memcpy(&valueBits, &doubleValue, sizeof(valueBits));
        value = uniqueExecutor->AllocScalar(&valueBits, op::DataType::DT_INT64);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
        self = inputView;
        out = outputView;
    }

    if (index->IsEmpty()) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
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
    selfContiguous = needUnsqueeze ? l0op::UnsqueezeNd(selfContiguous, squeezeDim, uniqueExecutor.get()) :
                                     selfContiguous;
    CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "squeeze failed!");
    needUnsqueeze = (indexContiguous->GetViewShape().GetDimNum() == 0);
    indexContiguous = needUnsqueeze ? l0op::UnsqueezeNd(indexContiguous, squeezeDim, uniqueExecutor.get()) :
                                      indexContiguous;
    CHECK_COND(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "squeeze failed!");
    // 将value 转为和self同类型的Tensor
    auto valueTensor = uniqueExecutor.get()->ConvertToTensor(value, selfContiguous->GetDataType());
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 1字节和8字节类型使用FrontQ分支，尾轴场景先转置到首轴。
    bool needTranspose = NeedTranspose(selfContiguous, dim);
    if (needTranspose) {
        selfContiguous = SwapDim(selfContiguous, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dim = 0;
    }

    const aclTensor* indexFillOut = nullptr;
    if (isInplace && Ops::NN::AclnnUtil::IsRegbase()) {
        indexFillOut = l0op::InplaceIndexFill(selfContiguous, indexContiguous, valueTensor, dim, uniqueExecutor.get());
        CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 进行IndexFill计算
        indexFillOut = l0op::IndexFill(selfContiguous, indexContiguous, valueTensor, dim, uniqueExecutor.get());
        CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

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

aclnnStatus aclnnIndexFillGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                           const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnIndexFill, DFX_IN(self, dim, index, value), DFX_OUT(out));
    return ExecIndexFillGetWorkspaceSize(self, dim, index, value, out, false, workspaceSize, executor);
}

aclnnStatus aclnnIndexFill(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexFill);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                  const aclScalar* value, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceIndexFill, DFX_IN(selfRef, dim, index, value), DFX_OUT(selfRef));
    return ExecIndexFillGetWorkspaceSize(selfRef, dim, index, value, selfRef, true, workspaceSize, executor);
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
