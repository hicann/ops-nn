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
 * \file aclnn_max_unpool3d.cpp
 * \brief
 */

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "index/scatter_elements_v2/op_host/op_api/scatter_elements.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_max_unpool3d.h"
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
#include "op_api/op_api_def.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t CDHW_DIM_NUM = 4;
static constexpr size_t NCDHW_DIM_NUM = 5;
static constexpr int64_t AXIS = 2;
static constexpr int64_t EXPECT_SIZE = 3;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize,
                                const aclIntArray* stride, const aclIntArray* padding, const aclTensor* outRef) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(outputSize, return false);
  OP_CHECK_NULL(stride, return false);
  OP_CHECK_NULL(padding, return false);
  OP_CHECK_NULL(outRef, return false);
  return true;
}

static inline bool CheckOutContiguous(const aclTensor* outRef) {
  OP_CHECK(op::IsContiguous(outRef), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outRef must be contiguous"), return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* indices, const aclTensor* outRef) {
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(outRef, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* indices) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  OP_CHECK(selfDimNum == CDHW_DIM_NUM || selfDimNum == NCDHW_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input to max_unpooling3d should be a 4d or 5d Tensor, "
                   "but got a tensor with dim %zu", selfDimNum), return false);

  OP_CHECK(selfDimNum == indices->GetViewShape().GetDimNum(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected shape of indices to be: %zu, but got: %zu",
                   selfDimNum, indices->GetViewShape().GetDimNum()), return false);

  OP_CHECK_SHAPE_NOT_EQUAL(self, indices, return false);
  return true;
}

static inline bool CheckOutShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* outRef) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  size_t outDimNum = outRef->GetViewShape().GetDimNum();
  OP_CHECK(selfDimNum == outDimNum,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out should be same in dimensions with self, but self is %zu, outRef is %zu",
                   selfDimNum, outDimNum), return false);

  int64_t dimN = 1;
  int64_t dimC = self->GetViewShape().GetDim(DIM_ZERO);
  int64_t outN = 1;
  int64_t outC = outRef->GetViewShape().GetDim(DIM_ZERO);
  int64_t outD = outRef->GetViewShape().GetDim(DIM_ONE);
  int64_t outH = outRef->GetViewShape().GetDim(DIM_TWO);
  int64_t outW = outRef->GetViewShape().GetDim(DIM_THREE);
  if (selfDimNum == NCDHW_DIM_NUM) {
    dimN = self->GetViewShape().GetDim(DIM_ZERO);
    dimC = self->GetViewShape().GetDim(DIM_ONE);
    outN = outRef->GetViewShape().GetDim(DIM_ZERO);
    outC = outRef->GetViewShape().GetDim(DIM_ONE);
    outD = outRef->GetViewShape().GetDim(DIM_TWO);
    outH = outRef->GetViewShape().GetDim(DIM_THREE);
    outW = outRef->GetViewShape().GetDim(DIM_FOUR);
  }
  OP_CHECK(dimN == outN && dimC == outC,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Out should be same in N、C dimensions with self, but self N: [%ld]、C:[%ld], "
                   "outRef N: [%ld]、C: [%ld]", dimN, dimC, outN, outC), return false);
  OP_CHECK(outD == (*outputSize)[DIM_ZERO] && outH == (*outputSize)[DIM_ONE] && outW == (*outputSize)[DIM_TWO],
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out should be same in D、H、W dimensions with outputSize, "
           "but outputSize D: [%ld]、H:[%ld]、W:[%ld], outRef D: [%ld]、H: [%ld]、W:[%ld]",
           (*outputSize)[DIM_ZERO], (*outputSize)[DIM_ONE], (*outputSize)[DIM_TWO], outD, outH, outW), return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const aclTensor* indices,
                                      const aclIntArray* outputSize, const aclIntArray* stride,
                                      const aclIntArray* padding, const aclTensor* outRef) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, indices, outputSize, stride, padding, outRef), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查out是否连续张量
  CHECK_RET(CheckOutContiguous(outRef), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, indices, outRef), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入的self和indices的shape
  CHECK_RET(CheckShape(self, indices), ACLNN_ERR_PARAM_INVALID);

  // 5. 校验输入self的元素值是否小于0
  CHECK_RET(CheckInpuNullTensorMaxUnPool3D(self), ACLNN_ERR_PARAM_INVALID);

  // 6. 检查输入的outputSize, stride, padding的size大小及元素值大小
  CHECK_RET(CheckIntArrayShapeMaxUnPool3D(self, outputSize, stride, padding), ACLNN_ERR_PARAM_INVALID);

  // 7. 校验out的shape是否合法
  CHECK_RET(CheckOutShape(self, outputSize, outRef), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

const aclIntArray *CalcMaxUnpool3dInputNewShape(int64_t dimN, int64_t dimC, int64_t dimD, int64_t dimH, int64_t dimW,
                                                aclOpExecutor *executor) {
  FVector<int64_t> newShape;
  newShape.emplace_back(dimN);
  newShape.emplace_back(dimC);
  newShape.emplace_back(dimD * dimH * dimW);
  return executor->AllocIntArray(newShape.data(), newShape.size());
}

aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor* self, const aclTensor* indices,
                                             const aclIntArray* outputSize, const aclIntArray* stride,
                                             const aclIntArray* padding, aclTensor* outRef,
                                             uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMaxUnpool3d, DFX_IN(self, indices, outputSize, stride, padding), DFX_OUT(outRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
           return ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, indices, outputSize, stride, padding, outRef);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  int64_t dimN = self->GetViewShape().GetDim(DIM_ZERO);
  int64_t dimC = 1;
  int64_t dimD = self->GetViewShape().GetDim(DIM_ONE);
  int64_t dimH = self->GetViewShape().GetDim(DIM_TWO);
  int64_t dimW = self->GetViewShape().GetDim(DIM_THREE);

  if (self->GetViewShape().GetDimNum() == NCDHW_DIM_NUM) {
    dimC = self->GetViewShape().GetDim(DIM_ONE);
    dimD = self->GetViewShape().GetDim(DIM_TWO);
    dimH = self->GetViewShape().GetDim(DIM_THREE);
    dimW = self->GetViewShape().GetDim(DIM_FOUR);
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outRefContiguous = l0op::Contiguous(outRef, uniqueExecutor.get());
  CHECK_RET(outRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto inputNewShapeArray = CalcMaxUnpool3dInputNewShape(dimN, dimC, dimD, dimH, dimW, uniqueExecutor.get());
  CHECK_RET(inputNewShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outNewShapeArray = CalcMaxUnpool3dInputNewShape(dimN, dimC, (*outputSize)[DIM_ZERO], (*outputSize)[DIM_ONE],
                                                       (*outputSize)[DIM_TWO], uniqueExecutor.get());
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

  auto outRefReshape2 = l0op::Reshape(scatterRes, outRef->GetViewShape(), uniqueExecutor.get());
  CHECK_RET(outRefReshape2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(outRefReshape2, outRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxUnpool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                             aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaxUnpool3d);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

