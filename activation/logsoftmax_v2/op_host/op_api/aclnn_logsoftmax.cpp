/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_logsoftmax.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "logsoftmax_v2.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void) {
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  if (!CheckSocVersionIsSupportBf16() &&
      ((self->GetDataType() == op::DataType::DT_BF16) || (out->GetDataType() == op::DataType::DT_BF16))) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "aclnnLogSoftmax is not support bfloat16 in current socversion.");
      return false;
  }
  // 检查self的数据类型是否在logsoftmax算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  // 检查out的数据类型是否在logsoftmax算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *out, op::DataType promoteType) {
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s can not cast to promote dtype %s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(DataType::DT_UNDEFINED).GetString());
    return false;
  }

  // 检查self的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, self->GetDataType(), return false);
  // 检查out的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

static bool CheckDim(const aclTensor *self, int64_t dim) {
  int64_t inputShapeLen = self->GetViewShape().GetDimNum();
  if (inputShapeLen == 0) {
    inputShapeLen++;
  }

  if ((dim < ((-1) * inputShapeLen)) || (dim > (inputShapeLen - 1))) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "provided dim %ld not in the range of input size %ld.", dim, inputShapeLen);
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  return true;
}

static aclnnStatus CheckLogSoftmaxParams(const aclTensor* self, int64_t dim, const aclTensor* out) {
  // 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 检查dim是否在[-r, r - 1]范围内，r为self的阶数
  CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 检查self和out能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, out, out->GetDataType()), ACLNN_ERR_PARAM_INVALID);

  // 检查shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnLogSoftmax, DFX_IN(self, dim), DFX_OUT(out));

  // 参数检查
  auto ret = CheckLogSoftmaxParams(self, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  op::DataType outType = out->GetDataType();

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型
  auto selfCasted = l0op::Cast(selfContiguous, outType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用LogSoftmaxV2算子kernel
  auto LogSoftmaxCal = l0op::LogSoftmaxV2(selfCasted, dim, uniqueExecutor.get());
  CHECK_RET(LogSoftmaxCal != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(LogSoftmaxCal, outType, uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSoftmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLogSoftmax);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
