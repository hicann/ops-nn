/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "hardtanh_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_hardtanh_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclScalar* min,
                         const aclScalar* max, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(out, return false);
  OP_CHECK_NULL(min, return false);
  OP_CHECK_NULL(max, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                            const aclTensor* out) {
  auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
  // 检查gradOutput的数据类型是否在HardtanhGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);

  // 检查self的数据类型是否在HardtanhGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

  // 检查gradOutput和self是否dtype一致
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);

  return true;
}

static bool CheckShapeValid(const aclTensor* gradOutput, const aclTensor* self) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  // 检查gradOutput和self是否shape一致
  OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, self, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclScalar* min,
                               const aclScalar* max, const aclTensor* out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, min, max, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查是否shape一致
  CHECK_RET(CheckShapeValid(gradOutput, self), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                  const aclScalar* min, const aclScalar* max, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnHardtanhBackward, DFX_IN(gradOutput, self, min, max), DFX_OUT(out));

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, min, max, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 将min转为float32
  float minValue = min->ToFloat();

  // 将min转为float32
  float maxValue = max->ToFloat();

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // HardtanhGrad算子的空tensor在kernel中支持
  if (gradOutput->IsEmpty() || self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutputContiguous转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入out转换成连续的tensor
  auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
  CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用HardtanhGrad算子kernel
  auto hardtanhBackwardOpOut = l0op::HardtanhGrad(gradOutputContiguous, selfContiguous,
                                                  minValue, maxValue,
                                                  outContiguous, uniqueExecutor.get());
  CHECK_RET(hardtanhBackwardOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(hardtanhBackwardOpOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardtanhBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHardtanhBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif