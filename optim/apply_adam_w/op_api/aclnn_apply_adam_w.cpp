/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_apply_adam_w.h"
#include "aclnn_kernels/contiguous.h"
#include "apply_adam_w.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListFromSocVersion() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  switch (curArch) {
    case NpuArch::DAV_2201: {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_3510: {
      return ASCEND950_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_1001: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
    default: {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
  }
}

static bool CheckNotNull(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                         const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                         const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                         const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional,
                         bool amsgrad) {
  OP_CHECK_NULL(varRef, return false);
  OP_CHECK_NULL(mRef, return false);
  OP_CHECK_NULL(vRef, return false);
  OP_CHECK_NULL(beta1Power, return false);
  OP_CHECK_NULL(beta2Power, return false);
  OP_CHECK_NULL(lr, return false);
  OP_CHECK_NULL(weightDecay, return false);
  OP_CHECK_NULL(beta1, return false);
  OP_CHECK_NULL(beta2, return false);
  OP_CHECK_NULL(eps, return false);
  OP_CHECK_NULL(grad, return false);
  if (amsgrad) {
    OP_CHECK_NULL(maxGradNormOptional, return false);
  }
  return true;
}

static bool CheckDatatype(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                          const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                          const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                          const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional) {
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListFromSocVersion();
  OP_CHECK_DTYPE_NOT_SUPPORT(varRef, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(mRef, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(vRef, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(beta1Power, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(beta2Power, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(lr, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(weightDecay, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(beta1, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(beta2, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(eps, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(grad, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, mRef, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, vRef, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, beta1Power, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, beta2Power, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, lr, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, weightDecay, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, beta1, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, beta2, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, eps, return false);
  OP_CHECK_DTYPE_NOT_SAME(varRef, grad, return false);
  if (maxGradNormOptional != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(maxGradNormOptional, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SAME(varRef, maxGradNormOptional, return false);
  }
  return true;
}

static bool CheckShape(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                       const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                       const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                       const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional) {
  OP_CHECK_SHAPE_NOT_EQUAL(mRef, varRef, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(vRef, varRef, return false);
  if (maxGradNormOptional != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(maxGradNormOptional, varRef, return false);
  }
  OP_CHECK_SHAPE_NOT_EQUAL(grad, varRef, return false);
  if (beta1Power->Numel() != 1 || beta2Power->Numel() != 1 || lr->Numel() != 1 || weightDecay->Numel() != 1 || 
      beta1->Numel() != 1 || beta2->Numel() != 1 || eps->Numel() != 1){
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                               const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                               const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                               const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional,
                               bool amsgrad) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
                         maxGradNormOptional, amsgrad), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDatatype(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
                         maxGradNormOptional), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的shape是否满足约束
  CHECK_RET(CheckShape(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
                       maxGradNormOptional), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef,
                                            const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr,
                                            const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2,
                                            const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional,
                                            bool amsgrad, bool maximize,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnApplyAdamW, DFX_IN(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
                                         maxGradNormOptional, amsgrad, maximize), DFX_OUT());
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(varRef, mRef, vRef, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad,
                         maxGradNormOptional, amsgrad);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor场景处理
  if (varRef->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入var转换成连续的tensor
  auto varContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
  CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入m转换成连续的tensor
  auto mContiguous = l0op::Contiguous(mRef, uniqueExecutor.get());
  CHECK_RET(mContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入v转换成连续的tensor
  auto vContiguous = l0op::Contiguous(vRef, uniqueExecutor.get());
  CHECK_RET(vContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入maxGradNorm转换成连续的tensor
  auto maxGradNormContiguous = maxGradNormOptional == nullptr ? nullptr :
    l0op::Contiguous(maxGradNormOptional, uniqueExecutor.get());
  CHECK_RET(maxGradNormOptional == nullptr || maxGradNormContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入grad转换成连续的tensor
  auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
  CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，调用ApplyAdamW算子完成计算
  auto [varOut, mOut, vOut] = l0op::ApplyAdamW(varContiguous, mContiguous, vContiguous, beta1Power, beta2Power,
                                               lr, weightDecay, beta1, beta2, eps, gradContiguous,
                                               maxGradNormContiguous, amsgrad, maximize, uniqueExecutor.get());
  CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(mOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(vOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，将计算结果拷贝到输出上，输出可能是非连续的tensor
  if (!IsContiguous(varRef)) {
    auto viewCopyResult = l0op::ViewCopy(varOut, varRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (!IsContiguous(mRef)) {
    auto viewCopyResult = l0op::ViewCopy(mOut, mRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (!IsContiguous(vRef)) {
    auto viewCopyResult = l0op::ViewCopy(vOut, vRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyAdamW(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnApplyAdamW);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif