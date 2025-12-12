/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "softshrink_grad.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_softshrink_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910 ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  }
}

static inline bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *lambd,
                         const aclTensor *gradInput) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(lambd, return false);
  OP_CHECK_NULL(gradInput, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self,
                                   [[maybe_unused]] const aclScalar *lambd,
                                   const aclTensor *gradInput) {
  auto supportList = GetDtypeSupportList();

  // 检查gradOutput的数据类型是否在SoftshrinkGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);

  // 检查output的数据类型是否在SoftshrinkGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查gradInput的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, supportList, return false);

  return true;
}

static inline bool CheckShapeValid(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

  // 检查gradOutput和self是否满足broadcast
  OP_CHECK_BROADCAST(gradOutput, self, return false);

  // 检查gradInput的shape与self和gradOutput的broadcast结果是否一致
  op::Shape broadcastShape;
  BroadcastInferShape(self->GetViewShape(), gradOutput->GetViewShape(), broadcastShape);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, broadcastShape, return false);

  return true;
}

static inline aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *lambd,
                               const aclTensor *gradInput) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, lambd, gradInput), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, lambd, gradInput), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查是否shape一致
  CHECK_RET(CheckShapeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static std::tuple<const aclTensor *, const aclTensor *> BroadcastTo(const aclTensor *gradOutput, const aclTensor *self,
                                                                    aclOpExecutor *executor) {
  Shape broadcastShape;
  if (!BroadcastInferShape(self->GetViewShape(), gradOutput->GetViewShape(), broadcastShape)) {
    return std::tuple(nullptr, nullptr);
  }
  FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = ToShapeVector(broadcastShape);
  auto broadcastShapeArray = executor->AllocIntArray(broadcastDims.data(), broadcastDims.size());
  CHECK_RET(broadcastShapeArray != nullptr, std::tuple(nullptr, nullptr));

  self = l0op::BroadcastTo(self, broadcastShapeArray, executor);
  CHECK_RET(self != nullptr, std::tuple(nullptr, nullptr));

  gradOutput = l0op::BroadcastTo(gradOutput, broadcastShapeArray, executor);
  CHECK_RET(gradOutput != nullptr, std::tuple(nullptr, nullptr));

  return std::tie(gradOutput, self);
}

aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                    const aclScalar *lambd, aclTensor *gradInput,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSoftshrinkBackward, DFX_IN(gradOutput, self, lambd), DFX_OUT(gradInput));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, lambd, gradInput);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 支持空进空出
  if (gradOutput->IsEmpty() || gradInput->IsEmpty()) {
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

  // 如果是viewShape不同，则需要broadCast成shape相同的
  if (gradOutput->GetViewShape() != self->GetViewShape()) {
    auto broadCastResult = BroadcastTo(gradOutputContiguous, selfContiguous, uniqueExecutor.get());

    gradOutputContiguous = std::get<0>(broadCastResult);
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    selfContiguous = std::get<1>(broadCastResult);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 将min转为float类型
  float lambdValue = lambd->ToFloat();

  // 如果gradOutput和self的数据类型不同，那么其中一个是FLOAT16，另一个是FLOAT类型，需要把FLOAT16 Cast为FLOAT
  if (gradOutputContiguous->GetDataType() != selfContiguous->GetDataType()) {
    if (gradOutputContiguous->GetDataType() != DataType::DT_FLOAT) {
      gradOutputContiguous = l0op::Cast(gradOutputContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
      CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (selfContiguous->GetDataType() != DataType::DT_FLOAT) {
      selfContiguous = l0op::Cast(selfContiguous, DataType::DT_FLOAT, uniqueExecutor.get());
      CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }

  // 调用SoftshrinkGrad算子kernel
  auto softShrinkBackwardOpOut =
      l0op::SoftShrinkGrad(gradOutputContiguous, selfContiguous, lambdValue, uniqueExecutor.get());
  CHECK_RET(softShrinkBackwardOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(softShrinkBackwardOpOut, gradInput, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftshrinkBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSoftshrinkBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif