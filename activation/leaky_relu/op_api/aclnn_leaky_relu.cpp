/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_leaky_relu.h"
#include "leaky_relu.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclScalar *negativeSlope, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(negativeSlope, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查输入self的数据类型是否在LeakyRelu算子的支持列表内
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar *negativeSlope, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, negativeSlope, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入tensor的shape是否异常，输出和输入的shape是否相同
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeakyReluGetWorkspaceSize(const aclTensor *self, const aclScalar *negativeSlope, aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnLeakyRelu, DFX_IN(self, negativeSlope), DFX_OUT(out));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, negativeSlope, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 输入为空tensor时，输出也是空tensor
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入tensor转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用LeakyRelu算子kernel，完成计算
  auto output = l0op::LeakyRelu(selfContiguous, negativeSlope->ToFloat(), uniqueExecutor.get());
  CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(output, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto copyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceLeakyReluGetWorkspaceSize(aclTensor *selfRef, const aclScalar *negativeSlope,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor) {
  return aclnnLeakyReluGetWorkspaceSize(selfRef, negativeSlope, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnLeakyRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLeakyRelu);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLeakyRelu(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceLeakyRelu);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
