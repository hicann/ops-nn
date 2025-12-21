/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_unpool2d_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "gather_elements.h"
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

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static constexpr size_t CHW_DIM_NUM = 3;
static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                               op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                const aclIntArray* outputSize, const aclTensor* out) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(outputSize, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                   const aclTensor* out) {
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(gradOutput->GetDataType(), out->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                       const aclIntArray* outputSize, const aclTensor* out) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  OP_CHECK(selfDimNum == CHW_DIM_NUM || selfDimNum == NCHW_DIM_NUM,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self to be a 3d or 4d Tensor, instead got: %zu", selfDimNum),
           return false);

  OP_CHECK(selfDimNum == gradOutput->GetViewShape().GetDimNum(),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Expected dim of self and gradOutput to be equal, instead got self: %zu, gradOutput: %zu",
                   selfDimNum, gradOutput->GetViewShape().GetDimNum()),
           return false);

  int64_t n = selfDimNum == CHW_DIM_NUM ? 1 : self->GetViewShape().GetDim(0);
  int64_t c = selfDimNum == CHW_DIM_NUM ? self->GetViewShape().GetDim(0) : self->GetViewShape().GetDim(1);
  int64_t gradn = selfDimNum == CHW_DIM_NUM ? 1 : gradOutput->GetViewShape().GetDim(0);
  int64_t gradc =
      selfDimNum == CHW_DIM_NUM ? gradOutput->GetViewShape().GetDim(0) : gradOutput->GetViewShape().GetDim(1);

  if (selfDimNum == CHW_DIM_NUM) {
    OP_CHECK(c == gradc,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Expected (C) of self and gradOutput to be equal, instead got self: (%ld), gradOutput: (%ld)",
                     c, gradc),
             return false);
  } else {
    OP_CHECK(n == gradn && c == gradc,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Expected (N, C) of self and gradOutput to be equal, instead got self: (%ld, %ld), gradOutput: "
                     "(%ld, %ld)", n, c, gradn, gradc),
             return false);
  }

  OP_CHECK_SHAPE_NOT_EQUAL(self, indices, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  OP_CHECK(outputSize->Size() == 2,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected size of outputSize must be 2, but got: %lu", outputSize->Size()),
           return false);

  int64_t dimh = selfDimNum == CHW_DIM_NUM ? 1 : 2;
  int64_t dimw = selfDimNum == CHW_DIM_NUM ? 2 : 3;
  int64_t oheight = (*outputSize)[0];
  int64_t owidth = (*outputSize)[1];
  int64_t gradh = gradOutput->GetViewShape().GetDim(dimh);
  int64_t gradw = gradOutput->GetViewShape().GetDim(dimw);
  OP_CHECK(oheight == gradh && owidth == gradw,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Inconsistent gradOutput size. output height = %ld, output width = %ld, but got gradOutput: %ldx%ld",
                   oheight, owidth, gradh, gradw),
           return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices,
                                      const aclIntArray* outputSize, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_COND(CheckNotNull(gradOutput, self, indices, outputSize, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_COND(CheckDtypeValid(gradOutput, self, indices, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

  // 3. 检查输出输出shape
  CHECK_COND(CheckShape(gradOutput, self, indices, outputSize, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                     const aclTensor* indices, const aclIntArray* outputSize,
                                                     aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMaxUnpool2dBackward, DFX_IN(gradOutput, self, indices, outputSize), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
           return ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, indices, outputSize, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (gradOutput->IsEmpty() && !self->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput is empty tensor, self should be empty tensor.");
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  size_t selfDimNum = self->GetViewShape().GetDimNum();
  int64_t oHeight = (*outputSize)[0];
  int64_t oWeight = (*outputSize)[1];
  int64_t dimN = 0;
  int64_t dimC = 0;
  int64_t dimH = 0;
  int64_t dimW = 0;
  if (selfDimNum == CHW_DIM_NUM) {
    dimN = 1;
    dimC = self->GetViewShape().GetDim(DIM_ZERO);
    dimH = self->GetViewShape().GetDim(DIM_ONE);
    dimW = self->GetViewShape().GetDim(DIM_TWO);
  } else {
    dimN = self->GetViewShape().GetDim(DIM_ZERO);
    dimC = self->GetViewShape().GetDim(DIM_ONE);
    dimH = self->GetViewShape().GetDim(DIM_TWO);
    dimW = self->GetViewShape().GetDim(DIM_THREE);
  }

  auto gradContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  FVector<int64_t> gradNewShape = {dimN, dimC, oHeight * oWeight};
  auto gradReshape =
      l0op::Reshape(gradContiguous, uniqueExecutor.get()->AllocIntArray(gradNewShape.data(), gradNewShape.size()),
                    uniqueExecutor.get());
  CHECK_RET(gradReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  FVector<int64_t> indicesNewShape = {dimN, dimC, dimH * dimW};
  auto indicesReshape = l0op::Reshape(
      indicesContiguous, uniqueExecutor.get()->AllocIntArray(indicesNewShape.data(), indicesNewShape.size()),
      uniqueExecutor.get());
  CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t dim = 2;
  auto grad = l0op::GatherElements(gradReshape, dim, indicesReshape, uniqueExecutor.get());

  auto outReshape = l0op::Reshape(grad, out->GetViewShape(), uniqueExecutor.get());
  CHECK_RET(outReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(outReshape, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxUnpool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMaxUnpool2dBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
