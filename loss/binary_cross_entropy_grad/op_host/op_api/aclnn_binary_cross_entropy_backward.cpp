/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <climits>

#include "aclnn_binary_cross_entropy_backward.h"
#include "binary_cross_entropy_backward.h"
#include "level0/ones_like.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;

enum Reduction {
  None,
  Mean,
  Sum,
  END
};

static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT16,
  DataType::DT_FLOAT
};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT16,
  DataType::DT_FLOAT,
  DataType::DT_BF16
};

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, aclTensor *out) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

template<typename T, typename... Ts>
static bool CheckDimension(const T &t, const Ts &... args) {
  if constexpr (std::is_same_v<std::remove_const_t<T>, aclTensor *>) {
    if (t) {
      OP_CHECK_MAX_DIM(t, MAX_SUPPORT_DIMS_NUMS, return false);
    }
  }

  if constexpr (sizeof...(args) > 0) {
    return CheckDimension(args...);
  }
  return true;
}

static bool CheckPromoteType(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional,
    const aclTensor *out, op::DataType &promoteType) {
  // 检查self和other能否做数据类型推导
  promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and target dtype %s can not promote dtype.",
        op::ToString(self->GetDataType()).GetString(), op::ToString(target->GetDataType()).GetString());
    return false;
  }

  op::DataType savedPromoteType = promoteType;
  promoteType = op::PromoteType(promoteType, gradOutput->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s and target dtype %s can not promote dtype.",
        op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(savedPromoteType).GetString());
    return false;
  }

  if (weightOptional) {
    savedPromoteType = promoteType;
    promoteType = op::PromoteType(promoteType, weightOptional->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight dtype %s and target dtype %s can not promote dtype.",
          op::ToString(weightOptional->GetDataType()).GetString(), op::ToString(savedPromoteType).GetString());
      return false;
    }
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional, aclTensor *out) {
  const auto& supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  if (weightOptional) {
    OP_CHECK_DTYPE_NOT_SUPPORT(weightOptional, supportList, return false);
  }
  return true;
}

static bool CheckFormat(const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional) {
  // self、target的数据格式必须相同
  if (self->GetStorageFormat() != target->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and target can't be different, self [%s], target [%s].",
        op::ToString(self->GetStorageFormat()).GetString(), op::ToString(target->GetStorageFormat()).GetString());
    return false;
  }

  // self、weightOptional的数据格式必须相同
  if (weightOptional && self->GetStorageFormat() != weightOptional->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Format of self and weightOptional can't be different, self [%s], weightOptional [%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(weightOptional->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional, aclTensor *out) {
  if (self->IsEmpty()) {
    return true;
  }

  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
  // gradOutput、target和weightOptional应能broadcast到self
  const op::Shape &selfShape = self->GetViewShape();
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, broadcastShape, return false);
  CHECK_RET(selfShape == broadcastShape, false);

  if (weightOptional) {
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, weightOptional, broadcastShape, return false);
    CHECK_RET(selfShape == broadcastShape, false);
  }

  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return false);
  CHECK_RET(selfShape == broadcastShape, false);
  return true;
}

static bool CheckReduction(int64_t reduction) {
  if (reduction > Sum || reduction < None) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but current is %ld", reduction);
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target,
    const aclTensor *weightOptional, int64_t reduction, aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的tensor维度是否小于等于8
  CHECK_RET(CheckDimension(gradOutput, self, target, weightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据API定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, target, weightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, target, weightOptional), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查shape是否满足约束
  CHECK_RET(CheckShape(gradOutput, self, target, weightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 6. 检查reduction是否合法
  CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    int64_t reduction,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBinaryCrossEntropyBackward,
      DFX_IN(gradOutput, self, target, weightOptional, reduction),
      DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, target, weightOptional, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 检查self能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
  op::DataType promoteType;
  CHECK_RET(CheckPromoteType(gradOutput, self, target, weightOptional, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // gradOutput如果非连续，需要转连续
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // target如果非连续，需要转连续
  auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // weightOptional如果为空，按self的shape创建全1的tensor
  const aclTensor *weightTensor;
  if (!weightOptional) {
    weightTensor = l0op::OnesLike(selfContiguous, uniqueExecutor.get());
  } else {
    weightTensor = l0op::Contiguous(weightOptional, uniqueExecutor.get());
  }
  CHECK_RET(weightTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto weightCasted = l0op::Cast(weightTensor, promoteType, uniqueExecutor.get());
  CHECK_RET(weightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子BinaryCrossEntropyOut进行计算
  static const std::string reductionStr[] = {"none", "mean", "sum"};
  auto bceGrad = l0op::BinaryCrossEntropyGrad(gradOutputCasted, selfCasted, targetCasted,
      weightCasted, reductionStr[reduction], uniqueExecutor.get());
  CHECK_RET(bceGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto bceGradCasted = l0op::Cast(bceGrad, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(bceGradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reformatBceGradOut = l0op::ReFormat(bceGradCasted, out->GetViewFormat());
  CHECK_RET(reformatBceGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(reformatBceGradOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyBackward(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBinaryCrossEntropyBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
