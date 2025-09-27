/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "gather_elements.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_max_unpool3d_backward.h"
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
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t NDHW_DIM_NUM = 4;
static constexpr size_t NCDHW_DIM_NUM = 5;
static constexpr int64_t AXIS = 2;
static constexpr int64_t EXPECT_SIZE = 3;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static inline bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                const aclIntArray* outputSize, const aclIntArray* stride, const aclIntArray* padding,
                                const aclTensor* out) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(outputSize, return false);
  OP_CHECK_NULL(stride, return false);
  OP_CHECK_NULL(padding, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckOutContiguous(const aclTensor* out) {
  OP_CHECK(op::IsContiguous(out), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out must be contiguous"), return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                                   const aclTensor* indices, const aclTensor* out) {
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(self, gradOutput->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(out, gradOutput->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT64, return false);
  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* indices, const aclTensor* out) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  OP_CHECK(selfDimNum == NDHW_DIM_NUM || selfDimNum == NCDHW_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input to max_unpooling3d should be a 4d or 5d Tensor, "
                   "but got a tensor with dim %zu", selfDimNum), return false);

  OP_CHECK(selfDimNum == indices->GetViewShape().GetDimNum(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected shape of indices to be: %zu, but got: %zu",
                   selfDimNum, indices->GetViewShape().GetDimNum()), return false);

  OP_CHECK(selfDimNum == out->GetViewShape().GetDimNum(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected shape of out to be: %zu, but got: %zu",
                   selfDimNum, out->GetViewShape().GetDimNum()), return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, indices, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static inline bool CheckOutShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* gradOutput) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  size_t gradOutputDimNum = gradOutput->GetViewShape().GetDimNum();
  OP_CHECK_WRONG_DIMENSION(self, gradOutputDimNum, return false);
  int64_t oT = (*outputSize)[DIM_ZERO];
  int64_t oH = (*outputSize)[DIM_ONE];
  int64_t oW = (*outputSize)[DIM_TWO];
  int64_t dimN = self->GetViewShape().GetDim(DIM_ZERO);
  int64_t dimC = 1;

  int64_t gradOutputN = gradOutput->GetViewShape().GetDim(DIM_ZERO);
  int64_t gradOutputC = 1;
  int64_t gradOutputD = gradOutput->GetViewShape().GetDim(DIM_ONE);
  int64_t gradOutputH = gradOutput->GetViewShape().GetDim(DIM_TWO);
  int64_t gradOutputW = gradOutput->GetViewShape().GetDim(DIM_THREE);

  if (selfDimNum == NCDHW_DIM_NUM) {
    dimC = self->GetViewShape().GetDim(DIM_ONE);
    gradOutputC = gradOutput->GetViewShape().GetDim(DIM_ONE);
    gradOutputD = gradOutput->GetViewShape().GetDim(DIM_TWO);
    gradOutputH = gradOutput->GetViewShape().GetDim(DIM_THREE);
    gradOutputW = gradOutput->GetViewShape().GetDim(DIM_FOUR);
  }

  OP_CHECK(oT == gradOutputD && oH == gradOutputH && oW == gradOutputW,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Inconsistant gradOutput size. oT= %ld, oH= %ld, oW= %ld."
                   " gradOutput: %ld x %ld x %ld.", oT, oH, oW, gradOutputD, gradOutputH, gradOutputW), return false);
  OP_CHECK(selfDimNum == gradOutputDimNum && dimN == gradOutputN && dimC == gradOutputC,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput and input Tensors should have same number of dimensions and "
                   "also the same number of channels/slices. But gradOutput is %zu dimensions, %ld batch size, %ld channels/slices,"
                   " self is %zu dimensions , %ld batch size, %ld channels/slices.", gradOutputDimNum, gradOutputN,
                   gradOutputC, selfDimNum, dimN, dimC), return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                      const aclIntArray* outputSize, const aclIntArray* stride,
                                      const aclIntArray* padding, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_COND(CheckNotNull(gradOutput, self, indices, outputSize, stride, padding, out), ACLNN_ERR_PARAM_NULLPTR,
             "CheckNotNull failed!");

  // 2. 检查out是否连续张量
  CHECK_COND(CheckOutContiguous(out), ACLNN_ERR_PARAM_INVALID, "CheckOutContiguous failed!");

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_COND(CheckDtypeValid(gradOutput, self, indices, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

  // 4. 检查输入的self、indices和out的shape
  CHECK_COND(CheckShape(self, indices, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

  // 5. 校验输入self的元素值是否小于0
  CHECK_COND(CheckInpuNullTensorMaxUnPool3D(self), ACLNN_ERR_PARAM_INVALID, "CheckInpuNullTensor failed!");

  // 6. 检查输入的outputSize, stride, padding的size大小及元素值大小
  CHECK_COND(CheckIntArrayShapeMaxUnPool3D(self, outputSize, stride, padding), ACLNN_ERR_PARAM_INVALID,
             "CheckIntArrayShape failed!");

  // 7. 校验gradOutput的shape是否合法
  CHECK_COND(CheckOutShape(self, outputSize, gradOutput), ACLNN_ERR_PARAM_INVALID, "CheckOutShape failed!");
  return ACLNN_SUCCESS;
}

const aclIntArray *CalcMaxUnpool3dGradInputNewShape(const aclTensor *target, aclOpExecutor *executor) {
  FVector<int64_t> newShape;
  size_t dimNum = target->GetViewShape().GetDimNum();
  int64_t dimN = target->GetViewShape().GetDim(DIM_ZERO);
  int64_t dimC = 1;
  int64_t dimD = target->GetViewShape().GetDim(DIM_ONE);
  int64_t dimH = target->GetViewShape().GetDim(DIM_TWO);
  int64_t dimW = target->GetViewShape().GetDim(DIM_THREE);

  if (dimNum == NCDHW_DIM_NUM) {
    dimC = target->GetViewShape().GetDim(DIM_ONE);
    dimD = target->GetViewShape().GetDim(DIM_TWO);
    dimH = target->GetViewShape().GetDim(DIM_THREE);
    dimW = target->GetViewShape().GetDim(DIM_FOUR);
  }
  newShape.emplace_back(dimN);
  newShape.emplace_back(dimC);
  newShape.emplace_back(dimD * dimH * dimW);
  return executor->AllocIntArray(newShape.data(), newShape.size());
}

aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                     const aclTensor* indices, const aclIntArray* outputSize,
                                                     const aclIntArray* stride, const aclIntArray* padding,
                                                     aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMaxUnpool3dBackward,
                 DFX_IN(gradOutput, self, indices, outputSize, stride, padding), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
           return ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, indices, outputSize, stride, padding, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (gradOutput->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclIntArray* gradOutputNewShapeArray = CalcMaxUnpool3dGradInputNewShape(gradOutputContiguous,
                                                                                uniqueExecutor.get());
  CHECK_RET(gradOutputNewShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclIntArray* indicesNewShapeArray = CalcMaxUnpool3dGradInputNewShape(indicesContiguous,
                                                                             uniqueExecutor.get());
  CHECK_RET(indicesNewShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradOutputReshape = l0op::Reshape(gradOutputContiguous, gradOutputNewShapeArray, uniqueExecutor.get());
  CHECK_RET(gradOutputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesReshape = l0op::Reshape(indicesContiguous, indicesNewShapeArray, uniqueExecutor.get());
  CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gatherRes = l0op::GatherElements(gradOutputReshape, AXIS, indicesReshape, uniqueExecutor.get());
  CHECK_RET(gatherRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outReshape = l0op::Reshape(gatherRes, out->GetViewShape(), uniqueExecutor.get());
  CHECK_RET(outReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(outReshape, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxUnpool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                             aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaxUnpool3dBackward);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

