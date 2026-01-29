/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_threshold_backward.h"
#include "../../../relu_grad/op_host/op_api/relu_grad.h"
#include "threshold_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
float thresholdVal_ = 0.0;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BF16, op::DataType::DT_INT64};

static bool IsFloatEqual(float a, float b) {
  return std::abs(a - b) <= std::numeric_limits<float>::epsilon();
}

static bool CheckPtrValid(const aclTensor *gradOutput, const aclTensor *self) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
    if (socVersion == SocVersion::ASCEND950 && IsFloatEqual(thresholdVal_, 0.0)) {
      // relugrad 支持int64
      return ASCEND950_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
      return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    }
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  // 检查gradOutput和self数据类型是否在ThresholdBackward算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *self) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckPtrValid(gradOutput, self), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

  // 3.输入维度校验
  CHECK_RET(CheckShape(gradOutput, self), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                   const aclScalar *threshold, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnThresholdBackward, DFX_IN(gradOutput, self, threshold), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // 固定写法，参数检查
  CHECK_RET(threshold != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  thresholdVal_ = threshold->ToFloat();
  CHECK_RET(out != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  auto ret = CheckParams(gradOutput, self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // 校验输入shape是否可broadcast
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return ACLNN_ERR_PARAM_INVALID);
  // 校验输出shape是否与推导出的broadcastShape相符
  if (broadcastShape != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }

  // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty() || gradOutput->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutput转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用 ReluGrad or ThresholdGradV2D 算子kernel
  const aclTensor* opOut;
  if (IsFloatEqual(thresholdVal_, 0.0)) {
    opOut = l0op::ReluGrad(gradOutputContiguous, selfContiguous, uniqueExecutor.get());
  } else {
    opOut = l0op::ThresholdGradV2D(gradOutputContiguous, selfContiguous, thresholdVal_, uniqueExecutor.get());
  }

  CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(opOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnThresholdBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnThresholdBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif