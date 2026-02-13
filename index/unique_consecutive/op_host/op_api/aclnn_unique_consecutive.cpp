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
 * \file aclnn_unique_consecutive.cpp
 * \brief
 */
#include "unique_consecutive.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_unique_consecutive.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t NoneN = 1000;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,      op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32,     op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE,     op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,      op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32,     op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE,     op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static aclnnStatus CheckParams(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                               int64_t dim) {
  // 1. 检查输入输出是否为nullptr
  CHECK_RET(CheckNotNull4Tensor(self, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  CHECK_RET(CheckDtypeValid1In1OutMatch(self, valueOut, inverseOut, countsOut, dtypeSupportList), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据Shape
  CHECK_RET(CheckShapeNotlimitDim1In1Out(self), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查参数dim
  if (dim != NoneN) {
    CHECK_RET(CheckDimValueWithUnique(self, dim), ACLNN_ERR_PARAM_INVALID);
  }

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(const aclTensor* self, bool returnInverse, bool returnCounts,
                                                   int64_t dim, aclTensor* valueOut, aclTensor* inverseOut,
                                                   aclTensor* countsOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnUniqueConsecutive, DFX_IN(self, returnInverse, returnCounts, dim),
                 DFX_OUT(valueOut, inverseOut, countsOut));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, valueOut, inverseOut, countsOut, dim);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (dim != NoneN) {
    auto valueViewShape = valueOut->GetViewShape();
    valueOut->SetStorageShape(valueViewShape);
    valueOut->SetOriginalShape(valueViewShape);
  }

  auto inverseViewShape = inverseOut->GetViewShape();
  inverseOut->SetStorageShape(inverseViewShape);
  inverseOut->SetOriginalShape(inverseViewShape);

  // 调用UniqueConsecutive算子
  auto opRet = l0op::UniqueConsecutive(selfContiguous, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut,
                          uniqueExecutor.get());
  CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueConsecutive(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnUniqueConsecutive);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
