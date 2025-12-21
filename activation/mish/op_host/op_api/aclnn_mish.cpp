/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/level2_base.h"
#include "mish.h"
#include "aclnn_mish.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
  // 检查self的数据类型是否在支持列表内
  auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  // 检查self和out的数据类型是否相等
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  // 输入和输出的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMishGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnMish, DFX_IN(self), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转换
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子进行计算
  auto mishResult = l0op::Mish(selfContiguous, uniqueExecutor.get());
  CHECK_RET(mishResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(mishResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMish(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMish);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceMishGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceMish, DFX_IN(selfRef), DFX_OUT(selfRef));
  return aclnnMishGetWorkspaceSize(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceMish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceMish);
  // 调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif