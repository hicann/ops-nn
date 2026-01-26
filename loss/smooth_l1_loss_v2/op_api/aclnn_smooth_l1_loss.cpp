/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_smooth_l1_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "smooth_l1_loss.h"
#include "level0/broadcast_to.h"
#include "level0/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;
static const size_t NC1HWC0_DIM = 5;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                              DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static inline bool IsAscendRegBase(void) {
  return Ops::NN::AclnnUtil::IsRegbase();
}

static bool CheckPromoteType(const op::DataType selfDtype, const op::DataType targetDtype,
                             const op::DataType resultDtype, op::DataType promoteType) {
  // 检查self和target能否做数据类型推导
  OP_CHECK(promoteType != DataType::DT_UNDEFINED,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and target dtype %s can not promote dtype.",
                   op::ToString(selfDtype).GetString(), op::ToString(targetDtype).GetString()),
           return false);

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, resultDtype, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *target) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在add算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查target的数据类型是否在add算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);

  return true;
}

static bool CheckReduction(int64_t reduction)
{
    // 检查self和other能否做数据类型推导
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but current is %ld.", reduction);
        return false;
    }
    return true;
}

static bool CheckNotNullTensor(const aclTensor *self, const aclTensor *target, const aclTensor *result) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(target, return false);
  OP_CHECK_NULL(result, return false);
  return true;
}

static bool CheckNeed5hdFormat(const aclTensor *tensor, int64_t reduction) {
  return ((tensor->GetStorageFormat() == Format::FORMAT_NCHW || tensor->GetStorageFormat() == Format::FORMAT_NHWC) &&
          reduction == 0) && !IsAscendRegBase();
}

static void CheckFormat(const aclTensor* self)
{
    // 检查self的format类型，NZ添加告警
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW(
            "Format of self gets [%s], this format may lead to precision failure.",
            op::ToString(self->GetStorageFormat()).GetString());
    }
}

static bool CheckShape(const aclTensor* self,const aclTensor* target, int64_t reduction,
                       aclTensor* result){
  //tensor维度不超过8维
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(target, MAX_SUPPORT_DIMS_NUMS, return false);

  op::Shape BroadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, BroadcastShape, return false);
  if (reduction==0 && BroadcastShape != result->GetViewShape()){
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of result should be %s, but current is %s.",
            op::ToString(BroadcastShape).GetString(), op::ToString(result->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *target, 
                               int64_t reduction, aclTensor *result) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNullTensor(self, target, result), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, target), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和target能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  op::DataType promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
  CHECK_RET(CheckPromoteType(self->GetDataType(), target->GetDataType(), result->GetDataType(), promoteType),
            ACLNN_ERR_PARAM_INVALID);

  // 4. 检查reduction是否符合规则
  CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);
  
  // 5. 检查tensor的shape是否合理
  CHECK_RET(CheckShape(self, target, reduction, result), ACLNN_ERR_PARAM_INVALID);

  // 6. 检查tensor的format是否合理，nz添加warning打印
  CheckFormat(self);

  return ACLNN_SUCCESS;
}

static const aclTensor *InputPreProcess(const aclTensor *input, int64_t reduction,
                                        op::DataType promoteType, aclOpExecutor *executor) {
  auto inputContiguous = l0op::Contiguous(input, executor);
  OP_CHECK(inputContiguous != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Contiguous return nullptr."), return nullptr);

  auto inputCast = l0op::Cast(inputContiguous, promoteType, executor);
  OP_CHECK(inputCast != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The cast return nullptr."), return nullptr);

  if (CheckNeed5hdFormat(inputCast, reduction)) {
    inputCast = l0op::TransData(inputCast, Format::FORMAT_NC1HWC0, 1, executor);
    OP_CHECK(inputCast != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The TransData return nullptr."),
             return nullptr);
  } else if (inputCast->GetStorageFormat() != Format::FORMAT_ND) {
    inputCast = l0op::ReFormat(inputCast, Format::FORMAT_ND);
    OP_CHECK(inputCast != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReFormat return nullptr."),
             return nullptr);
  }
  return inputCast;
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

static aclnnStatus CheckShapeBroadcast(const aclTensor** self, const aclTensor** target, aclOpExecutor *executor)
{
  // 判断输入shape不相等需要调用BroadcastTo
  if ((*self)->GetViewShape() != (*target)->GetViewShape()) {
    op::Shape broadcastShape;
    if (BroadcastInferShape((*self)->GetViewShape(), (*target)->GetViewShape(), broadcastShape)) {
      op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
      auto broadcastShapeArray = executor->AllocIntArray(broadcastDims.data(), broadcastDims.size());
      CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
      *self = l0op::BroadcastTo(*self, broadcastShapeArray, executor);
      CHECK_RET(*self != nullptr, ACLNN_ERR_INNER_NULLPTR);
      *target = l0op::BroadcastTo(*target, broadcastShapeArray, executor);
      CHECK_RET(*target != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
  }
  return ACLNN_SUCCESS;
}

inline static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor) {
  FVector<int64_t> tmp = {1};
  auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), DataType::DT_INT64);
  auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());
  CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

  FVector<float> valVector = {val};
  auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
  auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
  CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

inline static aclnnStatus EmptyTensorCompute(int64_t reduction, aclTensor* out, aclOpExecutor* executor) {
  aclnnStatus ret;
  if (reduction == 0) {
    ret = ACLNN_SUCCESS;
  } else if (reduction == 1) {
    ret = FillScalar(out, NAN, executor);
  } else {
    ret = FillScalar(out, 0, executor);
  }
  return ret;
}

aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction,
                                              float beta, aclTensor *result, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSmoothL1Loss, DFX_IN(self, target, reduction, beta), DFX_OUT(result));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, target, reduction, result);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // SmoothL1Loss算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty() || target->IsEmpty()) {
    // 根据实际支持情况补充
    ret = EmptyTensorCompute(reduction, result, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  ret = CheckShapeBroadcast(&self, &target, uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  auto promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
  auto selfpost = InputPreProcess(self, reduction, promoteType, uniqueExecutor.get());
  OP_CHECK(selfpost != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self with InputPreProcess return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  auto targetpost = InputPreProcess(target, reduction, promoteType, uniqueExecutor.get());
  OP_CHECK(targetpost != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The target with InputPreProcess return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  auto smoothL1LossResult =
      l0op::SmoothL1Loss(selfpost, targetpost, GetReductionStr(reduction), beta, uniqueExecutor.get());
  OP_CHECK(smoothL1LossResult != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The reslut with SmoothL1Loss return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);
  const aclTensor *smoothL1LossResultWithTrans = nullptr;
  if ((smoothL1LossResult->GetStorageShape().GetDimNum() == NC1HWC0_DIM &&
      (result->GetStorageFormat() == Format::FORMAT_NCHW || result->GetStorageFormat() == Format::FORMAT_NHWC))
     && !IsAscendRegBase()) {
    smoothL1LossResultWithTrans =
        l0op::TransData(smoothL1LossResult, GetPrimaryFormat(result->GetOriginalFormat()), 1, uniqueExecutor.get());
  } else {
    smoothL1LossResultWithTrans = l0op::ReFormat(smoothL1LossResult, result->GetStorageFormat());
  }
  OP_CHECK(smoothL1LossResultWithTrans != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The smoothL1LossResult with TransData return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto smoothL1LossResultCast = l0op::Cast(smoothL1LossResultWithTrans, result->GetDataType(), uniqueExecutor.get());
  OP_CHECK(smoothL1LossResultCast != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The smoothL1LossResult with cast return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(smoothL1LossResultCast, result, uniqueExecutor.get());
  OP_CHECK(viewCopyResult != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The reslut with ViewCopy return nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSmoothL1Loss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSmoothL1Loss);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
