/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_binary_cross_entropy_with_logits_backward.h"
#include "binary_cross_entropy_with_logits_backward.h"
#include "level0/ones_like.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;

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

#ifdef __cplusplus
extern "C" {
#endif

enum Reductions {
  None,
  Mean,
  Sum,
  END
};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_WITH_BF16 = {
  DataType::DT_FLOAT16,
  DataType::DT_FLOAT,
  DataType::DT_BF16
};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT16,
  DataType::DT_FLOAT
};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      Ops::NN::AclnnUtil::IsRegbase()) {
    return DTYPE_SUPPORT_LIST_WITH_BF16;
  }
  return DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, aclTensor *out) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckPromoteType(const aclTensor *gradOutput, const aclTensor *self,
    const aclTensor *target, const aclTensor *weightOptional, const aclTensor *posWeightOptional,
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

  if (posWeightOptional) {
    savedPromoteType = promoteType;
    promoteType = op::PromoteType(promoteType, posWeightOptional->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pos_weight dtype %s and target dtype %s can not promote dtype.",
          op::ToString(posWeightOptional->GetDataType()).GetString(), op::ToString(savedPromoteType).GetString());
      return false;
    }
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                            const aclTensor* weightOptional, const aclTensor* posWeightOptional, const aclTensor* out) {
  auto supportList = GetDtypeSupportList();
  // 检查gradOutput的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查target的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);

  // 检查weightOptional的数据类型是否在支持列表内，weightOptional为空例外
  if (weightOptional != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(weightOptional, supportList, return false);
  }

  // 检查posWeightOptional的数据类型是否在支持列表内，posWeightOptional为空例外
  if (posWeightOptional != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(posWeightOptional, supportList, return false);
  }

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  // 输出类型与target类型一致
  OP_CHECK_DTYPE_NOT_MATCH(out, target->GetDataType(), return false);

  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *target, const aclTensor* out) {
  // self、target的数据格式必须相同
  if (self->GetStorageFormat() != target->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and target can't be different, self [%s], target [%s].",
        op::ToString(self->GetStorageFormat()).GetString(), op::ToString(target->GetStorageFormat()).GetString());
    return false;
  }

  // self、out的数据格式必须相同
  if (self->GetStorageFormat() != out->GetStorageFormat()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of self and out can't be different, self [%s], out [%s].",
        op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                       const aclTensor* weightOptional, const aclTensor* posWeightOptional, const aclTensor* out) {
  if (self->IsEmpty()) {
    return true;
  }

  // target与self, out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, target, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

  // gradOutput、weightOptional应能broadcast到self
  const op::Shape& selfShape = self->GetViewShape();
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, broadcastShape, return false);
  CHECK_RET(selfShape == broadcastShape, false);

  if (weightOptional) {
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(weightOptional, self, broadcastShape, return false);
    CHECK_RET(selfShape == broadcastShape, false);
  }

  if (posWeightOptional) {
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(posWeightOptional, self, broadcastShape, return false);
    CHECK_RET(selfShape == broadcastShape, false);
  }

  return true;
}

static bool CheckReduction(int64_t reduction) {
  if (reduction > Sum || reduction < None) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but current is %ld", reduction);
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                               const aclTensor* weightOptional, const aclTensor* posWeightOptional, int64_t reduction,
                               aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的tensor维度是否小于等于8
  CHECK_RET(CheckDimension(gradOutput, self, target, weightOptional, posWeightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据API定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, self, target, weightOptional, posWeightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查shape是否满足约束
  CHECK_RET(CheckShape(gradOutput, self, target, weightOptional, posWeightOptional, out), ACLNN_ERR_PARAM_INVALID);

  // 6. 检查reduction是否合法
  CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus BinaryCrossEntropyWithLogitsBackwardStub(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *posWeightOptional,
    int64_t reduction,
    aclTensor *out,
    aclOpExecutor *executor) {
  // 检查self能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
  op::DataType promoteType;
  CHECK_RET(CheckPromoteType(gradOutput, self, target, weightOptional, posWeightOptional, out, promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // gradOutput如果非连续，需要转连续
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, executor);
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // target如果非连续，需要转连续
  auto targetContiguous = l0op::Contiguous(target, executor);
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // weightOptional如果为空，按self的shape创建全1的tensor
  auto weightTensor = (weightOptional == nullptr) ? l0op::OnesLike(selfContiguous, executor)
                                                  : l0op::Contiguous(weightOptional, executor);
  CHECK_RET(weightTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // weightOptional如果为空，按self的shape创建全1的tensor
  auto posWeightTensor = (posWeightOptional == nullptr) ? l0op::OnesLike(selfContiguous, executor)
                                                        : l0op::Contiguous(posWeightOptional, executor);
  CHECK_RET(posWeightTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, executor);
  CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor);
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto targetCasted = l0op::Cast(targetContiguous, promoteType, executor);
  CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto weightCasted = l0op::Cast(weightTensor, promoteType, executor);
  CHECK_RET(weightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto posWeightCasted = l0op::Cast(posWeightTensor, promoteType, executor);
  CHECK_RET(posWeightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子SigmoidCrossEntropyWithLogitsGradV2进行计算
  static const std::string reductionStrs[] = {"none", "mean", "sum"};
  auto bceWithLogitsGrad = l0op::SigmoidCrossEntropyWithLogitsGradV2(
      gradOutputCasted, selfCasted, targetCasted, weightCasted, posWeightCasted, reductionStrs[reduction], executor);
  CHECK_RET(bceWithLogitsGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto bceWithLogitsGradCasted = l0op::Cast(bceWithLogitsGrad, out->GetDataType(), executor);
  CHECK_RET(bceWithLogitsGradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reformatBceWithLogitsGradOut = l0op::ReFormat(bceWithLogitsGradCasted, out->GetViewFormat());
  CHECK_RET(reformatBceWithLogitsGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(reformatBceWithLogitsGradOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *posWeightOptional,
    int64_t reduction,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBinaryCrossEntropyWithLogitsBackward,
      DFX_IN(gradOutput, self, target, weightOptional, posWeightOptional, reduction),
      DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, self, target, weightOptional, posWeightOptional, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 开始计算
  ret = BinaryCrossEntropyWithLogitsBackwardStub(gradOutput, self, target, weightOptional, posWeightOptional,
                                                      reduction, out, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackward(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBinaryCrossEntropyWithLogitsBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
