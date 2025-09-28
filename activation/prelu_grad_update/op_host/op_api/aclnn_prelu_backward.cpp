/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prelu_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "prelu_grad_update.h"
#include "../../../prelu_grad_reduce/op_host/op_api/prelu_grad_reduce.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT,
  DataType::DT_FLOAT16
};
static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT,
  DataType::DT_FLOAT16,
  DataType::DT_BF16
};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static inline bool CheckNotNull(const aclTensor *gradOutput,
                                const aclTensor *self,
                                const aclTensor *weight,
                                const aclTensor *gradInput,
                                const aclTensor *gradWeight) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(gradInput, return false);
  OP_CHECK_NULL(gradWeight, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *gradOutput,
                                   const aclTensor *self,
                                   const aclTensor *weight,
                                   const aclTensor *gradInput,
                                   const aclTensor *gradWeight) {
  const auto& supportList = GetDtypeSupportList();
  // 检查self类型是否支持
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  // 检查类型是否与self一致
  OP_CHECK_DTYPE_NOT_SAME(gradOutput, self, return false);
  OP_CHECK_DTYPE_NOT_SAME(weight, self, return false);
  OP_CHECK_DTYPE_NOT_SAME(gradInput, self, return false);
  OP_CHECK_DTYPE_NOT_SAME(gradWeight, self, return false);

  return true;
}

static inline bool CheckShape(const aclTensor *gradOutput,
                              const aclTensor *self,
                              const aclTensor *weight,
                              const aclTensor *gradInput,
                              const aclTensor *gradWeight) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(weight, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(gradWeight, MAX_SUPPORT_DIMS_NUMS, return false);

  OP_CHECK_SHAPE_NOT_EQUAL(gradInput, self, return false);

  // 检查weight的元素个数是否等于1或者self的通道数
  int64_t weightNum = weight->Numel();
  int64_t selfDimNum = self->GetViewShape().GetDimNum();
  if (weightNum != 1) {
    // 不支持0维Tensor
    OP_CHECK_MIN_DIM(self, 1, return false);
    // self为1维时, 视为1通道
    int64_t channelSize = (selfDimNum > 1) ? self->GetViewShape().GetDim(1) : 1;
    if (weightNum != channelSize){
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the number of weight elements must be 1 or self channel size");
      return false;
    }

    op::Shape expectGradWeightShape = {channelSize};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradWeight, expectGradWeightShape, return false);
    } else {
      OP_CHECK_SHAPE_NOT_EQUAL(weight, gradWeight, return false);
    }

  // check the shape for gradOutput and self
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return false);
  if (self->GetViewShape() != broadcastShape) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the broadcastShape[%s] of gradOutput and self must be equal with selfShape[%s].",
            op::ToString(broadcastShape).GetString(), op::ToString(self->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput,
                               const aclTensor *self,
                               const aclTensor *weight,
                               const aclTensor *gradInput,
                               const aclTensor *gradWeight) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, weight, gradInput, gradWeight), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型是否准确
  CHECK_RET(CheckDtypeValid(gradOutput, self, weight, gradInput, gradWeight), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape
  CHECK_RET(CheckShape(gradOutput, self, weight, gradInput, gradWeight), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreluBackwardGetWorkspaceSize(const aclTensor *gradOutput,
                                               const aclTensor *self,
                                               const aclTensor *weight,
                                               aclTensor *gradInput,
                                               aclTensor *gradWeight,
                                               uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPreluBackward, DFX_IN(gradOutput, self, weight), DFX_OUT(gradInput, gradWeight));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, weight, gradInput, gradWeight);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果gradOutput是空tensor，则gradInput也是空tensor，直接返回
  if (gradOutput->IsEmpty() || self->IsEmpty() || weight->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutput转换成连续的Tensor
  auto contiguousGradOutput = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(contiguousGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto originalSelfShape = contiguousSelf->GetViewShape();
  // self是一维时需要升维为(1, N)
  if (contiguousSelf->GetViewShape().GetDimNum() == 1){
    op::Shape newSelfShape = {1, contiguousSelf->Size()};
    contiguousSelf = l0op::Reshape(contiguousSelf, newSelfShape, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousWeight = l0op::Contiguous(weight, uniqueExecutor.get());
  CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // weight如果不是1维的需转成1维
  if (contiguousWeight->GetViewShape().GetDimNum() != 1){
    op::Shape newWeightShape = {-1};
    contiguousWeight = l0op::Reshape(contiguousWeight, newWeightShape, uniqueExecutor.get());
    CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  const aclTensor *preluGradInput = nullptr;
  const aclTensor *update = nullptr;

  // 调用PReluGradUpdate算子kernel
  auto preluGradUpdateOut = l0op::PReluGradUpdate(contiguousGradOutput, contiguousSelf, contiguousWeight,
                                      uniqueExecutor.get());
  preluGradInput = std::get<0>(preluGradUpdateOut);
  update = std::get<1>(preluGradUpdateOut);
  CHECK_RET(preluGradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(update != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用PReluGradReduce算子kernel
  auto preluGradReduceOut = l0op::PReluGradReduce(contiguousGradOutput, contiguousSelf, contiguousWeight,
                                                  update, uniqueExecutor.get());
  CHECK_RET(preluGradReduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (preluGradInput->GetViewShape() != originalSelfShape){
    preluGradInput = l0op::Reshape(preluGradInput, originalSelfShape, uniqueExecutor.get());
    CHECK_RET(preluGradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，将计算结果拷贝到输出gradInput上, gradInput可能是非连续的tensor
  auto gradInputViewCopyResult = l0op::ViewCopy(preluGradInput, gradInput, uniqueExecutor.get());
  CHECK_RET(gradInputViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // gradWeight的shape可能为[1, 1, 1] 而 preluGradReduceOut的shape为[1]
  if (preluGradReduceOut->GetViewShape() != gradWeight->GetViewShape()){
    preluGradReduceOut = l0op::Reshape(preluGradReduceOut, gradWeight->GetViewShape(), uniqueExecutor.get());
    CHECK_RET(preluGradReduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto gradWeightViewCopyResult = l0op::ViewCopy(preluGradReduceOut, gradWeight, uniqueExecutor.get());
  CHECK_RET(gradWeightViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                               aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPreluBackward);

  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
