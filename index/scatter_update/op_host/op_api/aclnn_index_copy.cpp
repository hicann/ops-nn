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
 * \file aclnn_index_copy.cpp
 * \brief
 */
#include "aclnn_index_copy.h"
#include "aclnn_kernels/transpose.h"
#include "scatter_update.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* inplace_index_copy / index_copy 算子的完整计算流程如下:
 * selfRef                           index                 source
 *   |                                  |                     |
 *   \                                  |                     |
 * Contiguous(workspace_0)    Contiguous(workspace_1)   Contiguous(workspace_2)
 *      \                               |                      |
 *     Transpose(workspace_3)           |           Transpose(workspace_4)
 *               \                      |                     /
 *                      ScatterNdUpdate(workspace_5)
 *                                      |
 *                              Transpose(workspace_6)
 *                                      |
 *                            ViewCopy(workspace_7)
 *                                      |
 *                                selfRef/outRef
 */

constexpr size_t MAX_DIM_LEN = 8;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_INT8,
  op::DataType::DT_UINT8, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL,
  // aicore
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_910B = {
  op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL,
  // aicore
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST_910_95 = {
  op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_COMPLEX64,
  op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL,
  // aicore
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_UINT32,
  op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT64, op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* index, const aclTensor* source,
                                const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(source, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return SELF_DTYPE_SUPPORT_LIST_910B;
  } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return SELF_DTYPE_SUPPORT_LIST_910_95;
  } else {
    return SELF_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* index,
                            const aclTensor* source, const aclTensor* out) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  // 检查index的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return false);
  // self和source的数据类型要一致
  OP_CHECK_DTYPE_NOT_SAME(self, source, return false);
  // self和out的数据类型要一致
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  // 输入输出的格式需要一致
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. self [%s], out [%s].",
            ToString(self->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static inline bool IsSliceShapeSame(const aclTensor* self, const aclTensor* source, int64_t dim) {
  auto selfShape = self->GetViewShape();
  auto sourceShape = source->GetViewShape();

  size_t dimNum = selfShape.GetDimNum();
  int64_t posDim = dim < 0 ? dim + static_cast<int64_t>(dimNum) : dim;
  for (size_t idx = 0; idx < dimNum; idx++) {
    if (selfShape.GetDim(idx) != sourceShape.GetDim(idx) && idx != static_cast<uint64_t>(posDim)) {
      return false;
    }
  }
  return true;
}

static bool CheckShape(const aclTensor* self, int64_t dim, const aclTensor* index,
                      const aclTensor* source, const aclTensor* out) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(source, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(index, 1, return false);

  OP_CHECK_SHAPE_NOT_EQUAL(self, out , return false);

  auto selfShape = self->GetViewShape();
  auto sourceShape = source->GetViewShape();
  if (sourceShape.IsScalar() && index->GetViewShape().GetShapeSize() != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When source is scalar,index should have one element,but got %ld.", index->Size());
    return false;
  }

  if (selfShape.GetDimNum() != sourceShape.GetDimNum() && !selfShape.IsScalar() && !sourceShape.IsScalar()) {
    OP_LOGE(
        ACLNN_ERR_PARAM_INVALID,
        "When source and self are not scalars,their dimensionality must match,but got self dimensionality %s,source "
        "dimensionlity %s",
        op::ToString(selfShape).GetString(), op::ToString(sourceShape).GetString());
    return false;
  }

  // 检查参数dim是否合法
  int64_t tmpDim = static_cast<int64_t>(selfShape.GetDimNum());
  if (tmpDim == 0 && dim != 0 && dim != -1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-1, 0], but got %ld)",
            dim);
    return false;
  } else if (tmpDim > 0 && (dim < -tmpDim || dim >= tmpDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-%ld, %ld],"
            "but got %ld)", tmpDim, tmpDim - 1, dim);
    return false;
  }

  if (!IsSliceShapeSame(self, source, dim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self and source must have same slice shapes");
    return false;
  }

  return true;
}

static inline bool CheckElementNum(const aclTensor* index, const aclTensor* source, int64_t dim) {
  dim = dim >= 0 ? dim : dim + source->GetViewShape().GetDimNum();
  auto dimSize = source->GetViewShape().GetDim(dim);
  if (index->GetViewShape().GetShapeSize() != dimSize && !source->GetViewShape().IsScalar()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The element num of index (%ld) should be equal to source size in dimension %ld, actual is %ld",
            index->GetViewShape().GetShapeSize(), dim, dimSize);
    return false;
  }

  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, int64_t dim, const aclTensor* index,
                                      const aclTensor* source, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, source, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, index, source, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入形状是否满足
  CHECK_RET(CheckShape(self, dim, index, source, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入元素数量是否满足
  CHECK_RET(CheckElementNum(index, source, dim), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查输入输出format是否一致
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor* TransposeBySpecifiedAxis(const aclTensor* self, int64_t axis, aclOpExecutor* executor) {
  auto dimSize = (int64_t)(self->GetViewShape().GetDimNum());
  std::vector<int64_t> perm(dimSize);

  for (int64_t i = 0; i < dimSize; i++) {
    perm[i] = i;
  }

  std::swap(perm[axis], perm[0]);
  auto valuePerm = executor->AllocIntArray(perm.data(), dimSize);
  OP_CHECK(valuePerm != nullptr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "selfTransposed is nullptr"), return nullptr);

  auto selfTransposed = l0op::Transpose(self, valuePerm, executor);
  OP_CHECK(selfTransposed != nullptr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "selfTransposed is nullptr"), return nullptr);

  return selfTransposed;
}

aclnnStatus ExecIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                          const aclTensor* source, aclTensor* outRef,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(selfRef, dim, index, source, outRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (selfRef->IsEmpty() || index->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入selfRef转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入index转换成连续的tensor
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入source转换成连续的tensor
  auto sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
  CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  op::Shape rowVector;
  rowVector.AppendDim(-1);
  // 当前scatterupdate aicpu算子不支持传入0维tensor，需要做reshape处理
  const aclTensor* selfRefReShape = nullptr;
  if (selfRef->GetViewShape().IsScalar()) {
    dim = 0;
    selfRefReShape = l0op::Reshape(selfContiguous, rowVector, uniqueExecutor.get());
  } else {
    dim = dim >= 0 ? dim : dim + selfRef->GetViewShape().GetDimNum();
    selfRefReShape = selfContiguous;
  }

  const aclTensor* indexReShape = nullptr;
  if (index->GetViewShape().IsScalar()) {
    indexReShape = l0op::Reshape(indexContiguous, rowVector, uniqueExecutor.get());
  } else {
    indexReShape = indexContiguous;
  }

  const aclTensor* sourceReShape = nullptr;
  if (source->GetViewShape().IsScalar()) {
    sourceReShape = l0op::Reshape(sourceContiguous, rowVector, uniqueExecutor.get());
  } else {
    sourceReShape = sourceContiguous;
  }

  // 当前scatterupdate算子不支持指定dim做copy，需要使用transpose
  const aclTensor* kernelOut = nullptr;
  if (dim == 0) {
    // 调用ScatterUpdate算子
    kernelOut = l0op::ScatterUpdate(selfRefReShape, indexReShape, sourceReShape, uniqueExecutor.get(), false);
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 先做transpose再调用ScatterUpdate算子
    auto selfTransposed = TransposeBySpecifiedAxis(selfRefReShape, dim, uniqueExecutor.get());
    auto sourceTransposed = TransposeBySpecifiedAxis(sourceReShape, dim, uniqueExecutor.get());

    auto transposedKernelOut =
        l0op::ScatterUpdate(selfTransposed, indexReShape, sourceTransposed, uniqueExecutor.get(), false);
    CHECK_RET(transposedKernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    kernelOut = TransposeBySpecifiedAxis(transposedKernelOut, dim, uniqueExecutor.get());
  }

  // 固定写法，将计算结果拷贝到输出outRef上
  auto viewCopyResult = l0op::ViewCopy(kernelOut, outRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                          const aclTensor* source, aclTensor* outRef,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnIndexCopy, DFX_IN(selfRef, dim, index, source), DFX_OUT(outRef));
  return ExecIndexCopyGetWorkspaceSize(selfRef, dim, index, source, outRef, workspaceSize, executor);
}

aclnnStatus aclnnIndexCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnIndexCopy);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index,
                                                  const aclTensor* source, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceIndexCopy, DFX_IN(selfRef, dim, index, source), DFX_OUT(selfRef));
  return ExecIndexCopyGetWorkspaceSize(selfRef, dim, index, source, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceIndexCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceIndexCopy);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif