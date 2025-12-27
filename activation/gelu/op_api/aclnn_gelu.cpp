/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gelu.h"
#include "gelu.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of gelu does not support bfloat16 in current socversion.");
    return false;
  }

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out) {
  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(out->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s], out [%s]",
            ToString(self->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnGelu, DFX_IN(self), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子Gelu进行计算
  auto geluResult = l0op::Gelu(selfContiguous, uniqueExecutor.get());
  CHECK_RET(geluResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(geluResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGelu);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}