/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_soft_margin_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/broadcast_to.h"
#include "soft_margin_loss.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base_loss.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;
static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  const std::initializer_list<op::DataType> currentDtypeSupportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, currentDtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(target, currentDtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, currentDtypeSupportList, return false);
  return true;
}

// 检查self、target、out的shape是否满足broadcast规则
static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* out, int64_t reduction) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(target, MAX_SUPPORT_DIMS_NUMS, return false);

  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, broadcastShape, return false);

  // reduction为'none'，out的shape与broadcast的结果一致
  if (reduction == REDUCTION_NONE_NUM) {
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
  }

  // reduction为'mean'或'sum'，out为0-dimensional Tensor
  if (reduction != REDUCTION_NONE_NUM) {
    if (out->GetViewShape().GetDimNum() >= 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected a 0-dimensional tensor, but Shape of out is %s.",
              op::ToString(out->GetViewShape()).GetString());
      return false;
    }
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target, int64_t reduction,
                               const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull3Tensor(self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输出shape
  CHECK_RET(CheckShape(self, target, out, reduction), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus SoftMarginLossEmptyTensorCompute(int64_t reduction, aclTensor* out, aclOpExecutor* executor) {
  aclnnStatus ret = ACLNN_SUCCESS;
  // 1. reduction='none'，返回空tensor
  // 2. reduction='mean'，返回NAN
  // 3. reduction='sum'，返回0
  // 4. reduction其他值，返回0
  if (reduction == REDUCTION_MEAN_NUM) {
    ret = CheckFillScalarLoss(out, NAN, executor);
  } else if (reduction == REDUCTION_SUM_NUM) {
    ret = CheckFillScalarLoss(out, 0, executor);
  }
  return ret;
}

inline static const std::string &GetReductionStr(int64_t reduction) {
  if (reduction == REDUCTION_NONE_NUM) {
    return REDUCTION_NONE;
  } else if (reduction == REDUCTION_MEAN_NUM) {
    return REDUCTION_MEAN;
  }
  return REDUCTION_SUM;
}

aclnnStatus aclnnSoftMarginLossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSoftMarginLoss, DFX_IN(self, target, reduction), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, target, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || target->IsEmpty()) {
    // 根据实际支持情况补充
    ret = SoftMarginLossEmptyTensorCompute(reduction, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入target转换成连续的tensor
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 对self和target两个输入做隐式数据类型转换
  auto promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());

  // 将输入self的数据类型转换成隐式数据类型
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入target的数据类型转换成隐式数据类型
  auto targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfBroadcast = selfCasted;
  auto targetBroadcast = targetCasted;
  // 判断输入shape不相等需要调用BroadcastTo
  if (self->GetViewShape() != target->GetViewShape()) {
    op::Shape broadcastShape;
    if (BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
      op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
      auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
      CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      selfBroadcast = l0op::BroadcastTo(selfBroadcast, broadcastShapeArray, uniqueExecutor.get());
      CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      targetBroadcast = l0op::BroadcastTo(targetCasted, broadcastShapeArray, uniqueExecutor.get());
      CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  }

  // reduction为mean,sum的处理
  if (reduction != REDUCTION_NONE_NUM) {
    auto inputShape = selfBroadcast->GetViewShape();
    auto inputDims = inputShape.GetDimNum();
    int64_t shapes = 1;
    for (size_t i = 0; i < inputDims; i++) {
      shapes *= inputShape[i];
    }
    int64_t newShape[1] = {shapes};
    aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(newShape, 1);

    selfBroadcast = l0op::Reshape(selfBroadcast, shapeArray, uniqueExecutor.get());
    CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    targetBroadcast = l0op::Reshape(targetBroadcast, shapeArray, uniqueExecutor.get());
    CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 进行SoftMarginloss计算
  auto lossOut = l0op::SoftMarginLoss(selfBroadcast, targetBroadcast, GetReductionStr(reduction), uniqueExecutor.get());
  CHECK_RET(lossOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(lossOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftMarginLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSoftMarginLoss);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
