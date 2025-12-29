/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mse_loss_out.cpp
 * \brief
 */

#include "aclnn_mse_loss_out.h"
#include "mse_loss.h"
#include "level0/broadcast_to.h"
#include "level0/reduce_mean.h"
#include "level0/reduce_sum_op.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base_loss.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                 op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16,
                                                                                  op::DataType::DT_BF16};

static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* out,
                       int64_t reduction) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, broadcastShape, return false);
  if (reduction == REDUCTION_NONE_NUM) {
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
  } else {
    op::Shape reduceShape;
    for (size_t i = 1; i < broadcastShape.GetDimNum(); i++) {
      reduceShape.AppendDim(broadcastShape.GetDim(i));
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, reduceShape, return false);
  }
  return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target,
                                      int64_t reduction, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull3Tensor(self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValidMseLoss(self, target, out, ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查reduction是否符合规则
  CHECK_RET(CheckReductionMseLoss(reduction, REDUCTION_SUM_NUM, REDUCTION_NONE_NUM), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输出shape
  CHECK_RET(CheckShape(self, target, out, reduction), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

inline static const std::string &GetReductionStr(int64_t reduction) {
  if (reduction == REDUCTION_NONE_NUM) {
    return REDUCTION_NONE;
  } else if (reduction == REDUCTION_MEAN_NUM) {
    return REDUCTION_MEAN;
  } else {
    return REDUCTION_SUM;
  }
}

aclnnStatus aclnnMseLossOutGetWorkspaceSize(const aclTensor* self, const aclTensor* target,
                                            int64_t reduction, aclTensor* out,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMseLossOut, DFX_IN(self, target, reduction), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, target, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || target->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // MseLoss算子需要对self和target两个输入做隐式数据类型转换，根据具体算子语义按需调用
  auto promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入target转换成连续的tensor
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* selfBroadcast = nullptr;
  const aclTensor* targetBroadcast = nullptr;
  // 判断输入shape不相等需要调用BroadcastTo
  if (self->GetViewShape() != target->GetViewShape()) {
    op::Shape broadcastShape;
    if (BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
      op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
      auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
      CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      selfBroadcast = l0op::BroadcastTo(selfCasted, broadcastShapeArray, uniqueExecutor.get());
      CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      targetBroadcast = l0op::BroadcastTo(targetCasted, broadcastShapeArray, uniqueExecutor.get());
      CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  } else {
    selfBroadcast = selfCasted;
    targetBroadcast = targetCasted;
  }

  // 进行mseloss计算
  auto lossOut = l0op::MseLoss(selfBroadcast, targetBroadcast, GetReductionStr(0), uniqueExecutor.get());
  CHECK_RET(lossOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* loss;
  FVector<int64_t> tmp = {0};
  auto dim = uniqueExecutor.get()->AllocIntArray(tmp.data(), tmp.size());
  CHECK_RET(dim != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (reduction == REDUCTION_MEAN_NUM) {
    loss = l0op::ReduceMean(lossOut, dim, false, uniqueExecutor.get());
  } else if (reduction == REDUCTION_SUM_NUM) {
    loss = l0op::ReduceSumOp(lossOut, dim, false, uniqueExecutor.get());
  } else {
    loss = lossOut;
  }

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(loss, out->GetDataType(), uniqueExecutor.get());
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

aclnnStatus aclnnMseLossOut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMseLossOut);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
