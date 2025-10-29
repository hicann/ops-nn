/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_binary_cross_entropy_with_logits.h"
#include "binary_cross_entropy_with_logits.h"
#include "level0/fill.h"
#include "level0/ones_like.h"
#include "level0/reduce_mean.h"
#include "level0/reduce_sum_op.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t REDUCTION_MAX_NUM = 2;

enum Reduction { None = 0, Mean = 1, Sum = 2 };
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT,
  op::DataType::DT_FLOAT16
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT,
  op::DataType::DT_FLOAT16,
  op::DataType::DT_BF16
};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target, const aclTensor* out) {
  const auto& supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查target的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

  // 输出类型与target类型一致
  OP_CHECK_DTYPE_NOT_MATCH(target, out->GetDataType(), return false);

  return true;
}

static bool CheckReduction(int64_t reduction, const aclTensor* out) {
  // 如果reduction != None，out的shape应为Scalar
  if ((reduction != 0) && (out->GetViewShape().GetDimNum() != 0)) {
    OP_LOGW("Reduction is %ld, out shape must be [1].", reduction);
  }

  // 判断reduction取值是否合法
  if ((reduction < 0) || (reduction > REDUCTION_MAX_NUM)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction only support [0, 1, 2].");
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                       const aclTensor* posWeight) {
  // 所有算子的维度都不能超过8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(target, MAX_SUPPORT_DIMS_NUMS, return false);
  if (weight != nullptr) {
    OP_CHECK_MAX_DIM(weight, MAX_SUPPORT_DIMS_NUMS, return false);
  }
  if (posWeight != nullptr) {
    OP_CHECK_MAX_DIM(posWeight, MAX_SUPPORT_DIMS_NUMS, return false);
  }

  // target与self的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(self, target, return false);

  // weight和posWeight必须能broadcast到target的shape
  // 空tensor因为要单独输出特定值，不做broadcast判断，单独处理；与GPU行为保持一致
  if (target->IsEmpty()) {
    return true;
  }

  op::Shape broadcastShape;
  if (weight) {
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(weight, target, broadcastShape, return false);
  }

  if (posWeight) {
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(posWeight, target, broadcastShape, return false);
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                               const aclTensor* posWeight, int64_t reduction, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, target, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, target, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查reduction取值是否合法
  CHECK_RET(CheckReduction(reduction, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查tensor的shape是否合理
  CHECK_RET(CheckShape(self, target, weight, posWeight), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus HandleEmptyTensor(int64_t reduction, aclTensor* out,
                                     uint64_t* workspaceSize, UniqueExecutor& uniqueExecutor) {
  aclOpExecutor* executor = uniqueExecutor.get();

  // 如果reduction为None，则返回空tensor
  if (reduction == Reduction::None) {
    *workspaceSize = 0;
    return ACLNN_SUCCESS;
  }

  // 如果reduction为Mean，则返回填Nan tensor；
  // 如果reduction为Sum，则返回填0 tensor
  const aclScalar* valueScalar = nullptr;
  if (reduction == Reduction::Mean) {
    valueScalar = executor->AllocScalar(NAN);
  } else {
    valueScalar = executor->AllocScalar(0);
  }

  // 生成与outshape一致的结果为Nan/0的tensor作为输出tensor
  auto valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
  auto fillShape = op::ToShapeVector(out->GetOriginalShape());
  aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
  const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);

  auto binaryCrossEntropyOut = l0op::Fill(dims, valueTensor, shapeArray, executor);
  CHECK_RET(binaryCrossEntropyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上
  auto viewCopyResult = l0op::ViewCopy(binaryCrossEntropyOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();

  return ACLNN_SUCCESS;
}

static aclnnStatus HandleMeanAndSumReductionOut(int64_t reduction, const aclTensor* out,
                                                const aclTensor** reduceOut, aclOpExecutor* executor) {
  if (reduction == Reduction::None) {
    return ACLNN_SUCCESS;
  }

  // 计算维度，求所有维度的均值或者和
  FVector<int64_t> dimAll;
  op::Shape shapeOut = out->GetViewShape();
  for (size_t idx = 0; idx < shapeOut.GetDimNum(); idx++) {
      dimAll.push_back(static_cast<int64_t>(idx));
  }
  aclIntArray* axes = executor->AllocIntArray(dimAll.data(), dimAll.size());
  CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用ReduceMean或ReduceSum完成求均值或求和
  if (reduction == Reduction::Sum) {    // sum
    *reduceOut = l0op::ReduceSumOp(out, axes, false, executor);
  } else {   // mean
    *reduceOut = l0op::ReduceMean(out, axes, false, executor);
  }
  CHECK_RET(*reduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

static aclnnStatus BinaryCrossEntropyWithLogitsStub(const aclTensor* self, const aclTensor* target,
                                                    const aclTensor* weight, const aclTensor* posWeight,
                                                    int64_t reduction, aclTensor* out,
                                                    aclOpExecutor* executor) {
  // 连续性转换
  auto selfContiguous = l0op::Contiguous(self, executor);
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto targetContiguous = l0op::Contiguous(target, executor);
  CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 类型转换
  auto promoteType = PromoteType(selfContiguous->GetDataType(), targetContiguous->GetDataType());
  auto selfCast = l0op::Cast(selfContiguous, promoteType, executor);
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto targetCast = l0op::Cast(targetContiguous, promoteType, executor);
  CHECK_RET(targetCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 可选参数weight，如果没有定义，则用默认值；同时进行类型转换
  auto weightContiguous =
    ((weight == nullptr) ? l0op::OnesLike(selfCast, executor) : l0op::Contiguous(weight, executor));
  CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto weightCast = l0op::Cast(weightContiguous, promoteType, executor);
  CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 可选参数posWeight，如果没有定义，则用默认值；同时进行类型转换
  auto posWeightContiguous =
    ((posWeight == nullptr) ? l0op::OnesLike(selfCast, executor) : l0op::Contiguous(posWeight, executor));
  CHECK_RET(posWeightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto posWeightCast = l0op::Cast(posWeightContiguous, promoteType, executor);
  CHECK_RET(posWeightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用SigmoidCrossEntropyWithLogitsV2接口完成bceWithLogits计算
  // 该算子为融合算子，先求none情况下的返回值，再依据reduction求最终值
  static const std::string reductionCurr = "none";
  auto bceWithLogitsOut = l0op::SigmoidCrossEntropyWithLogitsV2(selfCast, targetCast, weightCast,
                                                                posWeightCast, reductionCurr, executor);
  CHECK_RET(bceWithLogitsOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 通过ReduceMean或ReduceSum计算reduction为mean或sum时的损失值
  auto bceWithLogitsReduceOut = bceWithLogitsOut;
  if (reduction != Reduction::None) {
    auto ret = HandleMeanAndSumReductionOut(reduction, bceWithLogitsOut, &bceWithLogitsReduceOut, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  } else {
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out,
                                                bceWithLogitsOut->GetStorageShape(),
                                                return ACLNN_ERR_PARAM_INVALID);
  }

  // 输出类型转换成out dtype类型
  auto bceWithLogitsOutCast = l0op::Cast(bceWithLogitsReduceOut, out->GetDataType(), executor);
  CHECK_RET(bceWithLogitsOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(bceWithLogitsOutCast, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* self, const aclTensor* target,
                                                              const aclTensor* weightOptioanl,
                                                              const aclTensor* posWeightOptioanl, int64_t reduction,
                                                              aclTensor* out, uint64_t* workspaceSize,
                                                              aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnBinaryCrossEntropyWithLogits,
                 DFX_IN(self, target, weightOptioanl, posWeightOptioanl, reduction),
                 DFX_OUT(out));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, target, weightOptioanl, posWeightOptioanl, reduction, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty()) {
    ret = HandleEmptyTensor(reduction, out, workspaceSize, uniqueExecutor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 开始bce with logits计算
  auto bceWithLogitsRet = BinaryCrossEntropyWithLogitsStub(self, target, weightOptioanl, posWeightOptioanl,
                                                           reduction, out, uniqueExecutor.get());
  CHECK_RET(bceWithLogitsRet == ACLNN_SUCCESS, bceWithLogitsRet);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBinaryCrossEntropyWithLogits(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBinaryCrossEntropyWithLogits);

  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
