/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_multilabel_margin_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "multilabel_margin_loss.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";
static const int64_t MAX_SELF_DIM_NUM = 2;
static const int RESULT_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> TARGET_DTYPE_SUPPORT_LIST = {DataType::DT_INT64,
                                                                              DataType::DT_INT32};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  }
  return ASCEND910_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* target, const aclTensor* out,
                         const aclTensor* isTarget) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);
  OP_CHECK_NULL(isTarget, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target,
                            const aclTensor* out, const aclTensor* isTarget) {
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(target, TARGET_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(isTarget, self->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* out,
                       const aclTensor* isTarget, int64_t reduction) {
  size_t selfDimNum = self->GetViewShape().GetDimNum();
  bool valid = (selfDimNum == 2 && self->GetViewShape().GetDim(1) != 0) ||
               (selfDimNum == 1 && self->GetViewShape().GetDim(0) != 0) || (selfDimNum == 0);
  OP_CHECK(valid,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Expected non-empty vector or matrix with optional 0-dim batch size, but got: %s",
                   op::ToString(self->GetViewShape()).GetString()),
           return false);
  if (selfDimNum <= 1) {
    size_t dim = selfDimNum == 0 ? 1 : self->GetViewShape().GetDim(0);
    OP_CHECK(
      target->GetViewShape().GetDimNum() <= 1 && target->GetViewShape().GetShapeSize() == dim,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inconsistent target size: %s for self of size: %s",
              op::ToString(target->GetViewShape()).GetString(),
              op::ToString(self->GetViewShape()).GetString()),
              return false);
  } else if (selfDimNum == MAX_SELF_DIM_NUM) {
    size_t nFrame = self->GetViewShape().GetDim(0);
    size_t dim = self->GetViewShape().GetDim(1);
    OP_CHECK(
      target->GetViewShape().GetDimNum() == 2 && target->GetViewShape().GetDim(0) == nFrame &&
      target->GetViewShape().GetDim(1) == dim,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inconsistent target size: %s for self of size: %s",
              op::ToString(target->GetViewShape()).GetString(),
              op::ToString(self->GetViewShape()).GetString()),
              return false);
  }
  if (reduction == 0 && selfDimNum == MAX_SELF_DIM_NUM) {
    OP_CHECK(out->GetViewShape().GetDimNum() == 1 &&
             out->GetViewShape().GetDim(0) == self->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expect out shape [%zu], but got: %s.",
                     self->GetViewShape().GetDim(0),
                     op::ToString(out->GetViewShape()).GetString()), return false);
  } else {
    OP_CHECK(out->GetViewShape().GetDimNum() <= 1 && out->GetViewShape().GetShapeSize() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected a single element out tensor, but got: %s",
                     op::ToString(out->GetViewShape()).GetString()),
             return false);
  }
  if (selfDimNum == MAX_SELF_DIM_NUM && self->GetViewShape().GetDim(0) == 0) {
    OP_CHECK(
      isTarget->GetViewShape().GetDimNum() <= 2 && isTarget->GetViewShape().GetShapeSize() == 0,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inconsistent isTarget size: %s for self of size: %s",
              op::ToString(isTarget->GetViewShape()).GetString(),
              op::ToString(self->GetViewShape()).GetString()), return false);
  } else {
    OP_CHECK_SHAPE_NOT_EQUAL(isTarget, target, return false);
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target,
                               int64_t reduction, const aclTensor* out,
                               const aclTensor* isTarget) {
  // 1. 检查参数是否为空指针
  CHECK_COND(CheckNotNull(self, target, out, isTarget), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_COND(CheckDtypeValid(self, target, out, isTarget), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

  // 4. 检查输出输出shape
  CHECK_COND(CheckShape(self, target, out, isTarget, reduction), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

  return ACLNN_SUCCESS;
}

static const std::string &GetReductionStr(int64_t reduction) {
  if (reduction == 0) {
    return REDUCTION_NONE;
  } else if (reduction == 1) {
    return REDUCTION_MEAN;
  } else {
    return REDUCTION_SUM;
  }
}

static bool CheckTupleNullptr(std::tuple<aclTensor*, aclTensor*> tensorTuple)
{
    if (std::tuple_size<decltype(tensorTuple)>::value != RESULT_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The length of tuple returned by MaxPoolWithArgmaxV1 is not 2.");
        return false;
    }

    return (std::get<0>(tensorTuple) != nullptr)
            && (std::get<1>(tensorTuple) != nullptr);
}

aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(const aclTensor* self, const aclTensor* target,
                                                      int64_t reduction, aclTensor* out,
                                                      aclTensor* isTarget,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMultilabelMarginLoss, DFX_IN(self, target, reduction), DFX_OUT(out, isTarget));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "uniqueExecutor cannot be null!");

  // 固定写法，参数检查
  auto ret = CheckParams(self, target, reduction, out, isTarget);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_COND(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "contiguous self failed!");

  int64_t squeezeDim = 0;
  auto selfReshape =
    self->GetViewShape().GetDimNum() == 0 ? l0op::UnsqueezeNd(selfContiguous,
                                                              squeezeDim,
                                                              uniqueExecutor.get()) : selfContiguous;
  CHECK_COND(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR, "unsqueezeNd self failed!");

  // 固定写法，将输入target转换成连续的tensor
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_COND(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "contiguous target failed!");

  auto targetCasted = l0op::Cast(targetContiguous, DataType::DT_INT32, uniqueExecutor.get());
  CHECK_COND(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast target failed!");

  auto targetReshape =
      target->GetViewShape().GetDimNum() == 0 ? l0op::UnsqueezeNd(targetCasted,
                                                                  squeezeDim,
                                                                  uniqueExecutor.get()) : targetCasted;
  CHECK_COND(targetReshape != nullptr, ACLNN_ERR_INNER_NULLPTR, "target unsqueezeNd failed!");
  // 固定写法，将输入weight转换成连续的tensor
  // 进行MultilabelMarginLoss计算
  auto lossOut = l0op::MultilabelMarginLoss(
    selfReshape, targetReshape, GetReductionStr(reduction), out->GetViewShape(), uniqueExecutor.get());
  CHECK_COND(CheckTupleNullptr(lossOut), ACLNN_ERR_INNER_NULLPTR, "MultilabelMarginLoss failed!");

  auto outLoss = std::get<0>(lossOut);
  auto outIsTarget = std::get<1>(lossOut);

  auto castOut = l0op::Cast(outLoss, out->GetDataType(), uniqueExecutor.get());
  CHECK_COND(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast out failed!");

  auto castIsTarget = l0op::Cast(outIsTarget, out->GetDataType(), uniqueExecutor.get());
  CHECK_COND(castIsTarget != nullptr, ACLNN_ERR_INNER_NULLPTR, "cast isTarget failed!");

  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_COND(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewcopy out failed!");
  auto viewCopyIsTarget = l0op::ViewCopy(castIsTarget, isTarget, uniqueExecutor.get());
  CHECK_COND(viewCopyIsTarget != nullptr, ACLNN_ERR_INNER_NULLPTR, "viewcopy isTarget failed!");

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultilabelMarginLoss(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMultilabelMarginLoss);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
