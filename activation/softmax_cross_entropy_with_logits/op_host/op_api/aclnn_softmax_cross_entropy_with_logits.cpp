/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_softmax_cross_entropy_with_logits.h"
#include "softmax_cross_entropy_with_logits.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t dim_2 = 2;
static const std::initializer_list<DataType> Ascend910_dtype_support_list = {op::DataType::DT_FLOAT16,
                                                                             op::DataType::DT_FLOAT};

static const std::initializer_list<DataType> Ascend910B_dtype_support_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return Ascend910B_dtype_support_list;
  }
  return Ascend910_dtype_support_list;
}

static bool CheckNotNull(const aclTensor* features, const aclTensor* labels, const aclTensor* loss, const aclTensor* backprop) {
  OP_CHECK_NULL(features, return false);
  OP_CHECK_NULL(labels, return false);
  OP_CHECK_NULL(loss, return false);
  OP_CHECK_NULL(backprop, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* features, const aclTensor* labels, const aclTensor* loss, const aclTensor* backprop) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(features, supportList, return false);

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(labels, supportList, return false);

  // 检查outGelu的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(loss, supportList, return false);
  // 检查backprop的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(backprop, supportList, return false);

  // 检查输入和输出的数据类型是否一致
  OP_CHECK_DTYPE_NOT_MATCH(loss, features->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(backprop, features->GetDataType(), return false);

  return true;
}

static bool CheckShape(const aclTensor* features, const aclTensor* labels, const aclTensor* loss, const aclTensor* backprop) {
  OP_CHECK_MAX_DIM(features, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(labels, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(loss, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(backprop, MAX_SUPPORT_DIMS_NUMS, return false);

  int64_t dimNum = features->GetViewShape().GetDimNum();
  int64_t dimNum_2 = labels->GetViewShape().GetDimNum();
  if(dimNum != dim_2 || dimNum_2 != dim_2){
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "dimNum is %ld, must be 2, but check failed.", dimNum);
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* features, const aclTensor* labels, const aclTensor* loss, const aclTensor* backprop) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(features, labels, loss, backprop), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(features, labels, loss, backprop), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(features, labels, loss, backprop), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus ExecSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, const aclTensor* labels, const aclTensor* loss, const aclTensor* backprop, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  // 固定写法，参数检查
  auto ret = CheckParams(features, labels, loss, backprop);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 空Tensor处理
  if (features->IsEmpty()) {
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  if (labels->IsEmpty()) {
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto featuresContiguous = l0op::Contiguous(features, uniqueExecutor.get());
  CHECK_RET(featuresContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto labelsContiguous = l0op::Contiguous(labels, uniqueExecutor.get());
  CHECK_RET(labelsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子GeGlu行计算
  auto SoftmaxCrossEntropyWithLogitsResult = l0op::SoftmaxCrossEntropyWithLogits(featuresContiguous, labelsContiguous, uniqueExecutor.get());
  const aclTensor *SoftmaxCrossEntropyOutResult = std::get<0>(SoftmaxCrossEntropyWithLogitsResult);
  const aclTensor *LogitsResult = std::get<1>(SoftmaxCrossEntropyWithLogitsResult);
  CHECK_RET(SoftmaxCrossEntropyOutResult != nullptr && LogitsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  int64_t dimNum = loss->GetViewShape().GetDimNum();
  int64_t dimNum_1 = SoftmaxCrossEntropyOutResult->GetViewShape().GetDimNum();
  auto viewCopyOutResult = l0op::ViewCopy(SoftmaxCrossEntropyOutResult, loss, uniqueExecutor.get());
  CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyLogitsOutResult = l0op::ViewCopy(LogitsResult, backprop, uniqueExecutor.get());
  CHECK_RET(viewCopyLogitsOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,
                                                aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnSoftmaxCrossEntropyWithLogits, DFX_IN(features, labels), DFX_OUT(loss, backprop));
  return ExecSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(features, labels, loss, backprop, workspaceSize, executor);
}

aclnnStatus aclnnSoftmaxCrossEntropyWithLogits(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnSoftmaxCrossEntropyWithLogits);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
