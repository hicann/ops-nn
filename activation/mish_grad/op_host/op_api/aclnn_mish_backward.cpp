/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_mish_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "mish_grad.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
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

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                 op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16,
                                                                                  op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion() {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  switch (socVersion) {
    case SocVersion::ASCEND910B:
    case SocVersion::ASCEND910_93: {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    default: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
  }
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(gradInput, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput) {
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), gradInput->GetDataType(), return false);

  std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST = GetDtypeSupportListBySocVersion();

  // 检查gradOutput的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, CURRENT_DTYPE_SUPPORT_LIST, return false);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);

  // 检查gradInput的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, CURRENT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);

  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, broadcastShape, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, gradInput), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输出输出shape
  CHECK_RET(CheckShape(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

// getBroadcastShape for l0op::BroadcastTo
static aclIntArray* GetShape(const op::Shape broadcastShape,
                             aclOpExecutor* executor) {
  int64_t tensorSize = static_cast<int64_t>(broadcastShape.GetDimNum());
  std::vector<int64_t> tensorShape(tensorSize);
  for (int i = 0; i < tensorSize; i++) {
    tensorShape[i] = broadcastShape[i];
  }
  return executor->AllocIntArray(tensorShape.data(), tensorSize);
}

// 如果输入tensor shape与braodcast后的shape不一致，在进行反向计算前，先进行broadcasto操作。
static const aclTensor* BroadcastTensor(const aclTensor* self,
                                        const op::Shape broadcastShape,
                                        aclOpExecutor* executor) {
  // 如果self的shape与broadcast的不一致，进行BroadcastTo
  if (self->GetViewShape() != broadcastShape) {
    auto broadcastShapeIntArray = GetShape(broadcastShape, executor);
    if (broadcastShapeIntArray != nullptr) {
      return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
    }
  }
  return self;
}

aclnnStatus aclnnMishBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput,
                                              uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMishBackward, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, gradInput);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || gradOutput->IsEmpty() || gradInput->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = op::PromoteType(self->GetDataType(), gradOutput->GetDataType());

  // 固定写法，将输入gradOutput转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入gradoutput的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // shape不一致进行broadcast
  op::Shape broadcastShape1;
  BroadcastInferShape(gradOutputCasted->GetViewShape(), selfCasted->GetViewShape(), broadcastShape1);

  // self不一致进行broadcast
  selfCasted = BroadcastTensor(selfCasted, broadcastShape1, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 判断gradOutput是否需要进行broadcast
  gradOutputCasted = BroadcastTensor(gradOutputCasted, broadcastShape1, uniqueExecutor.get());
  CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行计算
  auto grad = l0op::MishGrad(gradOutputCasted, selfCasted, uniqueExecutor.get());
  CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto castedRes = l0op::Cast(grad, gradInput->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castedRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castedRes, gradInput, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMishBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMishBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
