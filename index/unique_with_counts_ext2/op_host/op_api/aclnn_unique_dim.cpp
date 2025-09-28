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
 * \file aclnn_unique_dim.cpp
 * \brief
 */

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "unique_with_counts_ext2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_unique_dim.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,      op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32,     op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE,     op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,      op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32,     op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE,     op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}
static bool CheckNotNull(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut) {
  // 检查输入张量self不为nullptr
  OP_CHECK_NULL(self, return false);
  // 检查输出张量valueOut不为nullptr
  OP_CHECK_NULL(valueOut, return false);
  // 检查输出张量inverseOut不为nullptr
  OP_CHECK_NULL(inverseOut, return false);
  // 检查输出张量countsOut不为nullptr
  OP_CHECK_NULL(countsOut, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut) {
  auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
  // 检查self的数据类型是否在UniqueDim算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查inverseOut数据类型
  OP_CHECK_DTYPE_NOT_MATCH(inverseOut, op::DataType::DT_INT64, return false);

  // 检查countsOut数据类型
  OP_CHECK_DTYPE_NOT_MATCH(countsOut, op::DataType::DT_INT64, return false);

  // 检查self与valueOut数据类型一致
  OP_CHECK_DTYPE_NOT_MATCH(valueOut, self->GetDataType(), return false);

  // 检查inverseOut与countsOut数据类型一致
  OP_CHECK_DTYPE_NOT_MATCH(inverseOut, countsOut->GetDataType(), return false);
  return true;
}

static bool CheckShapeValid(const aclTensor* self) {
  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static bool CheckDimValue(const aclTensor* self, int64_t dim) {
  int64_t dimSize = self->GetViewShape().GetDimNum();
  int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
  int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
  if (dim > dimMax || dim < dimMin) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The param of dim must be in the range of [%ld, %ld], but got %ld.", dimMin, dimMax,
            dim);
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                               int64_t dim) {
  // 1. 检查输入输出是否为nullptr
  CHECK_RET(CheckNotNull(self, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型
  CHECK_RET(CheckDtypeValid(self, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据Shape
  CHECK_RET(CheckShapeValid(self), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查参数dim
  CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse,
                                           int64_t dim, aclTensor* valueOut, aclTensor* inverseOut,
                                           aclTensor* countsOut, uint64_t* workspaceSize,
                                           aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnUniqueDim, DFX_IN(self, sorted, returnInverse, dim),
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

  auto valueViewShape = valueOut->GetViewShape();
  valueOut->SetStorageShape(valueViewShape);
  valueOut->SetOriginalShape(valueViewShape);

  auto countsViewShape = countsOut->GetViewShape();
  countsOut->SetStorageShape(countsViewShape);
  countsOut->SetOriginalShape(countsViewShape);

  // 调用UniqueDim算子
  auto opRet = l0op::UniqueWithCountsExt2(selfContiguous, sorted, returnInverse, dim, valueOut, inverseOut, countsOut,
                                        uniqueExecutor.get());
  CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnUniqueDim);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
