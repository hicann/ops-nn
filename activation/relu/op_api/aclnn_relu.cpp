/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_relu.h"
#include "aclnn_kernels/contiguous.h"
#include "relu.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INCLUDE_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  std::string dtypeListString = op::ToString(DTYPE_SUPPORT_LIST).GetString();
  if (!CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST)) {
    bool support_bf16 = CheckSocVersionIsSupportBf16();
    if (support_bf16) {
        dtypeListString = op::ToString(DTYPE_SUPPORT_LIST_INCLUDE_BF16).GetString();
    }
    if (!support_bf16 || (support_bf16 && self->GetDataType() != op::DataType::DT_BF16)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s should be in dtype support list %s.",
              op::ToString(self->GetDataType()).GetString(), dtypeListString.c_str());
      return false;
    }
  }
  OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus ExecReluGetWorkspaceSize(const aclTensor *self, const aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // tensor空进空出或者self与out是同一指针且self数据类型是UINT8
  if (self->IsEmpty() || (self == out && self->GetDataType() == op::DataType::DT_UINT8)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reluOpOut = selfContiguous;
  // 如果self不是UINT8类型，需要调用relu计算
  if (self->GetDataType() != op::DataType::DT_UINT8) {
    // 调用relu进行计算
    reluOpOut = l0op::Relu(selfContiguous, uniqueExecutor.get());
  }
  CHECK_RET(reluOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(reluOpOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnReluGetWorkspaceSize(const aclTensor *self,
                                      const aclTensor *out,
                                      uint64_t *workspaceSize,
                                      aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnRelu, DFX_IN(self), DFX_OUT(out));
  return ExecReluGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnRelu);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceReluGetWorkspaceSize(aclTensor *selfRef,
                                             uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceRelu, DFX_IN(selfRef), DFX_OUT(selfRef));
  return ExecReluGetWorkspaceSize(selfRef, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceRelu(void *workspace, uint64_t workspaceSize,
                             aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceRelu);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
