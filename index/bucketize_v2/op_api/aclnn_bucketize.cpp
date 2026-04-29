/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/framework_op.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_util.h"
#include "bucketize_v2.h"
#include "level0/fill.h"
#include "aclnn_bucketize.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t DIM_LEN_BOUNDARY = 1;
static constexpr int64_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_BF16,    op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor *self, const aclTensor *boundaries, const aclTensor *out) 
{
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(boundaries, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *boundaries,
                             const aclTensor *out, const bool outInt32) 
{
  // 获取芯片支持的数据类型
  const std::initializer_list<op::DataType> dtypeSupportList = ASCEND950_AICORE_DTYPE_SUPPORT_LIST;
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(boundaries, dtypeSupportList, return false);

  op::DataType expectedOutDtype = outInt32 ? op::DataType::DT_INT32 : op::DataType::DT_INT64;
  if (out->GetDataType() != expectedOutDtype) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor out expected dtype is %s but found %s when outInt32 is %s.",
          op::ToString(expectedOutDtype).GetString(), op::ToString(out->GetDataType()).GetString(), 
          outInt32 ? "true" : "false");
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *boundaries, const aclTensor* out) 
{
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  OP_CHECK_WRONG_DIMENSION(boundaries, DIM_LEN_BOUNDARY, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *boundaries,
                                const bool outInt32, const aclTensor* out) 
{
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, boundaries, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, boundaries, out, outInt32), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self, boundaries, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static op::DataType CombineCategories(const op::DataType higher, const op::DataType lower) 
{
  if (IsFloatingType(higher)) {
    return higher;
  }
  if (higher == op::DataType::DT_BOOL || IsFloatingType(lower)) {
    return op::PromoteType(higher, lower);
  }
  if (higher != op::DataType::DT_UNDEFINED) {
    return higher;
  }
  return lower;
}

aclnnStatus aclnnBucketizeGetWorkspaceSize(const aclTensor *self, const aclTensor *boundaries, const bool outInt32,
                                            const bool right, aclTensor *out, uint64_t *workspaceSize, 
                                            aclOpExecutor **executor) 
{
  OP_CHECK_COMM_INPUT(workspaceSize, executor); 

  // 固定写法，参数检查
  L2_DFX_PHASE_1(aclnnBucketize, DFX_IN(self, boundaries, outInt32, right), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (!Ops::NN::AclnnUtil::IsRegbase()) {
    OP_LOGE(ACLNN_ERR_INNER, "BucketizeV2 Op not support this Soc Version!");
    return ACLNN_ERR_INNER;
  }

  // 固定写法，参数检查
  auto ret = CheckParams(self, boundaries, outInt32, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 算子的空tensor的支持情况
  if (self->Numel() == 0) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  if (boundaries->Numel() == 0) {
    auto outputValue = uniqueExecutor.get()->AllocScalar(0);
    op::DataType expectedOutDtype = outInt32 ? op::DataType::DT_INT32 : op::DataType::DT_INT64;
    auto outputValueTensor = uniqueExecutor.get()->ConvertToTensor(outputValue, expectedOutDtype);
    auto outputFillShape = op::ToShapeVector(self->GetViewShape());
    aclIntArray *outputShapeArray = uniqueExecutor.get()->AllocIntArray(outputFillShape.data(), outputFillShape.size());
    CHECK_RET(outputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *outputDims = uniqueExecutor.get()->ConvertToTensor(outputFillShape.data(),
                                                          outputFillShape.size(), op::DataType::DT_INT64);
    auto bucketizeOut = l0op::Fill(outputDims, outputValueTensor, outputShapeArray, uniqueExecutor.get());
    CHECK_RET(bucketizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(bucketizeOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto boundariesContiguous = l0op::Contiguous(boundaries, uniqueExecutor.get());
  CHECK_RET(boundariesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // self和boundaries输入类型支持不同，进行类型合并转换
  op::DataType promoteType = op::DataType::DT_UNDEFINED;
  if (self->GetViewShape().GetDimNum() == 0) {
    promoteType = CombineCategories(boundaries->GetDataType(), self->GetDataType());
  } else {
    promoteType = op::PromoteType(self->GetDataType(), boundaries->GetDataType());
  }
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto boundariesCasted = l0op::Cast(boundariesContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(boundariesCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto bucketizeOut = l0op::BucketizeV2(selfCasted, boundariesCasted, outInt32, right,
                                             uniqueExecutor.get());
  CHECK_RET(bucketizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  
  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(bucketizeOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBucketize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream) 
{
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnBucketize);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
