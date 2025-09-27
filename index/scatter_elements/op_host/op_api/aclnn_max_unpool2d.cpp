/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_max_unpool2d.cpp
 * \brief
 */

#include "aclnn_max_unpool2d.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "index/scatter_elements_v2/op_host/op_api/scatter_elements.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t CHW_DIM_NUM = 3;
static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t AXIS = 2;
static constexpr int64_t EXPECT_SIZE = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize,
                                const aclTensor* outRef) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(outputSize, return false);
  OP_CHECK_NULL(outRef, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* indices, const aclTensor* outRef) {
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(outRef, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* indices) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  OP_CHECK(selfDimNum == CHW_DIM_NUM || selfDimNum == NCHW_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self to be a 3d or 4d Tensor, instead got: %zu", selfDimNum),
           return false);

  OP_CHECK(selfDimNum == indices->GetViewShape().GetDimNum(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Expected dim of self and indices to be equal, instead got self: %zu, gradOutput: %zu",
                   selfDimNum, indices->GetViewShape().GetDimNum()),
           return false);

  OP_CHECK_SHAPE_NOT_EQUAL(self, indices, return false);
  return true;
}

static bool CheckInpuNullTensor(const aclTensor* self) {
  auto inputShape = self->GetViewShape();
  size_t dimNum = inputShape.GetDimNum();

  for (size_t i = 1; i < dimNum; ++i) {
    if (inputShape.GetDim(i) <= 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_unpool2d():expected input to have non-empty spatiak dimensions, "
              "but input has sizes %zu with dimension %zu being empty.", dimNum, i);
      return false;
    }
  }
  return true;
}

static bool CheckOutputSize(const aclTensor* self, const aclIntArray* outputSize) {
  uint64_t size = outputSize->Size();
  if (size != EXPECT_SIZE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize length should be 2, but now is %lu.", size);
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    if ((*outputSize)[i] <= 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize value should greater than 0, but the sizes of %zu is %ld.",
              i, (*outputSize)[i]);
      return false;
    }
  }
  int64_t dimH = self->GetViewShape().GetDim(DIM_ONE);
  int64_t dimW = self->GetViewShape().GetDim(DIM_TWO);
  if (self->GetViewShape().GetDimNum() == NCHW_DIM_NUM) {
    dimH = self->GetViewShape().GetDim(DIM_TWO);
    dimW = self->GetViewShape().GetDim(DIM_THREE);
  }
  OP_CHECK(((*outputSize)[0] * (*outputSize)[1]) >= (dimH * dimW),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The output volums are of size %ld x %ld, should greater than or equal to "
           "self of size %ld x %ld.", (*outputSize)[0], (*outputSize)[1], dimH, dimW),
           return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const aclTensor* indices,
                                      const aclIntArray* outputSize, const aclTensor* outRef) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, indices, outputSize, outRef), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, indices, outRef), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输出shape
  CHECK_RET(CheckShape(self, indices), ACLNN_ERR_PARAM_INVALID);

  // 4. 校验输入tensor是否为空
  CHECK_RET(CheckInpuNullTensor(self), ACLNN_ERR_PARAM_INVALID);

  // 5. 校验output_size是否合法
  CHECK_RET(CheckOutputSize(self, outputSize), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

const aclIntArray *CalcMaxUnpool2dInputNewShape(int64_t dimN, int64_t dimC, int64_t dimH, int64_t dimW,
                                                aclOpExecutor *executor) {
  FVector<int64_t> newShape;
  newShape.emplace_back(dimN);
  newShape.emplace_back(dimC);
  newShape.emplace_back(dimH * dimW);
  return executor->AllocIntArray(newShape.data(), newShape.size());
}

aclnnStatus aclnnMaxUnpool2dGetWorkspaceSize(const aclTensor* self, const aclTensor* indices,
                                             const aclIntArray* outputSize, aclTensor* outRef,
                                             uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMaxUnpool2d, DFX_IN(self, indices, outputSize), DFX_OUT(outRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
           return ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, indices, outputSize, outRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  size_t selfDimNum = self->GetViewShape().GetDimNum();
  int64_t outH = (*outputSize)[0];
  int64_t outW = (*outputSize)[1];
  int64_t dimN = 1;
  int64_t dimC = self->GetViewShape().GetDim(DIM_ZERO);
  int64_t dimH = self->GetViewShape().GetDim(DIM_ONE);
  int64_t dimW = self->GetViewShape().GetDim(DIM_TWO);

  if (selfDimNum == NCHW_DIM_NUM) {
    dimN = self->GetViewShape().GetDim(DIM_ZERO);
    dimC = self->GetViewShape().GetDim(DIM_ONE);
    dimH = self->GetViewShape().GetDim(DIM_TWO);
    dimW = self->GetViewShape().GetDim(DIM_THREE);
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outRefContiguous = l0op::Contiguous(outRef, uniqueExecutor.get());
  CHECK_RET(outRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclIntArray* inputNewShapeArray = CalcMaxUnpool2dInputNewShape(dimN, dimC, dimH, dimW, uniqueExecutor.get());
  CHECK_RET(inputNewShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclIntArray* outNewShapeArray = CalcMaxUnpool2dInputNewShape(dimN, dimC, outH, outW, uniqueExecutor.get());
  CHECK_RET(outNewShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfReshape = l0op::Reshape(selfContiguous, inputNewShapeArray, uniqueExecutor.get());
  CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesReshape = l0op::Reshape(indicesContiguous, inputNewShapeArray, uniqueExecutor.get());
  CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outRefReshape = l0op::Reshape(outRefContiguous, outNewShapeArray, uniqueExecutor.get());
  CHECK_RET(outRefReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto zeroOut = l0op::ZerosLike(outRefReshape, uniqueExecutor.get());
  CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  static const std::string reductionCurr = "none";
  auto scatterRes =
      l0op::ScatterElements(zeroOut, indicesReshape, selfReshape, AXIS, reductionCurr, uniqueExecutor.get());
  CHECK_RET(scatterRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outReshape2 = l0op::Reshape(scatterRes, outRef->GetViewShape(), uniqueExecutor.get());
  CHECK_RET(outReshape2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(outReshape2, outRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxUnpool2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                             aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaxUnpool2d);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

