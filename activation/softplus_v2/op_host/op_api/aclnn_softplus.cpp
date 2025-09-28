/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_softplus.h"
#include "softplus.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
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

/* Softplus 算子的完整计算流程如下:
 *                   self
 *                    |
 *          Contiguous(workspace_0)
 *                    |
 *          SoftplusV2(workspace_1)
 *                    |
 *            Cast(workspace_2)
 *                    |
 *                ViewCopy
 *                    |
 *                 result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclScalar *beta, const aclScalar *threshold,
                         const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(beta, return false);
  OP_CHECK_NULL(threshold, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtype(const aclTensor *self, const aclScalar *beta, const aclScalar *threshold,
                       const aclTensor *out) {
  auto DTYPE_SUPPORT_LIST = GetDtypeSupportListV1(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  // 检查self的数据类型是否在Softplus算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查beta能否转换为self的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(beta->GetDataType(), self->GetDataType(), return false);

  // 检查threshold能否转换为self的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(threshold->GetDataType(), self->GetDataType(), return false);

  // 检查self的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

  // 输入输出shape是否一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar *beta, const aclScalar *threshold,
                               const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, beta, threshold, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验；检查能否进行数据类型转化
  CHECK_RET(CheckDtype(self, beta, threshold, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入shape是否符合，输入输出shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftplusGetWorkspaceSize(const aclTensor *self, const aclScalar *beta, const aclScalar *threshold,
                                          aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSoftplus, DFX_IN(self, beta, threshold), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, beta, threshold, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // Softplus算子的空tensor在kernel中支持
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用SoftplusV2算子kernel
  auto softplusOpOut = l0op::SoftplusV2(selfContiguous, beta->ToFloat(),
                                        threshold->ToFloat(), uniqueExecutor.get());
  CHECK_RET(softplusOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(softplusOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftplus(void *workspace, uint64_t workspaceSize,
                          aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSoftplus);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif