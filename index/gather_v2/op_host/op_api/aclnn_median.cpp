/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_median.h"
#include "level0/median_dim.h"
#include "level0/add.h"
#include "level0/masked_fill.h"
#include "level0/reduce_min.h"
#include "index/gather_elements/op_host/op_api/gather_elements.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/select.h"
#include "level0/div.h"
#include "level0/right_shift.h"
#include "level0/zero_op.h"
#include "level0/equal.h"
#include "level0/not_equal.h"
#include "level0/fill.h"
#include "level0/gather_v2.h"
#include "level0/reduce_sum_op.h"
#include "aclnn_kernels/reshape.h"
#include "level0/sort.h"
#include "level0/sub.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int32_t MAX_INT32 = 2147483647;
static const int32_t MAX_CONVERT_NUM = 16777216;
static constexpr size_t DIM_ZERO = 0;
static const int64_t SMALLSORTLIMIT = 16;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_UINT8,
  op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_LIST_910B = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910 = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
  op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_LIST_910 = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return DTYPE_SUPPORT_LIST_910B;
  }
  return DTYPE_SUPPORT_LIST_910;
}

static inline const std::initializer_list<op::DataType>& GetFloatList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return FLOAT_DTYPE_LIST_910B;
  }
  return FLOAT_DTYPE_LIST_910;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *valuesOut) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(valuesOut, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *valuesOut) {
  auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(valuesOut, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_MATCH(valuesOut, self->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *valuesOut) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(valuesOut, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *valuesOut) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, valuesOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, valuesOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, valuesOut), ACLNN_ERR_PARAM_INVALID);

  // 4. 当self的数据类型为BFLOAT16时，self的最后一根轴大小不能等于1
  auto dimSize = self->GetViewShape().GetDimNum();
  if (dimSize != 0) {
    if (self->GetDataType() == op::DataType::DT_BF16 && self->GetViewShape().GetDim(dimSize - 1) == 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last axis value is not support 1 when input type is BF16.");
      return ACLNN_ERR_PARAM_INVALID;
    }
  }

  return ACLNN_SUCCESS;
}

// 检查tuple <values, indices>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleNotNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple)
{
  return (std::get<0>(tensorTuple)!=nullptr) && (std::get<1>(tensorTuple)!=nullptr);
}


// 检查dType符合预期
static bool CheckDtypeValidDim(const aclTensor *self, const aclTensor *valuesOut, const aclTensor *indicesOut)
{
  if (!CheckDtypeValid(self, valuesOut)) {
    return false;
  }

  if (indicesOut->GetDataType() != op::DataType::DT_INT64) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indicesOut dtype %s should be in dtype support list [%s].",
            op::ToString(indicesOut->GetDataType()).GetString(), "op::DataType::DT_INT64");
    return false;
  }

  return true;
}

static bool CheckNotNullDim(const aclTensor *self, const aclTensor *valuesOut, const aclTensor *indicesOut) {
  if (!CheckNotNull(self, valuesOut)) {
    return false;
  }

  if (indicesOut == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "indicesOut should not be null.");
    return false;
  }

  return true;
}

// 获得tensor的维度数
static inline int64_t GetTensorDim(const aclTensor *self)
{
  return static_cast<int64_t> (self->GetViewShape().GetDimNum());
}

// dim应该处于范围 [-N, N-1]中
static bool CheckDimValue(const aclTensor *self, const int64_t dim)
{
  int64_t dimSize = GetTensorDim(self);
  int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
  int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
  if ((dim > dimMax) || (dim < dimMin)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in range [%ld, %ld].", dimMin, dimMax);
    return false;
  }
  return true;
}

static bool CheckShapeDim(const aclTensor *self, int64_t dim, bool keepDim, const aclTensor *valuesOut,
                          const aclTensor *indicesOut) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(valuesOut, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(indicesOut, MAX_SUPPORT_DIMS_NUMS, return false);

  auto expectedDimNum = keepDim ? self->GetViewShape().GetDimNum() : self->GetViewShape().GetDimNum() - 1;
  if (self->GetViewShape().GetDimNum() == 0) {
    expectedDimNum = self->GetViewShape().GetDimNum();
  }
  OP_CHECK_WRONG_DIMENSION(valuesOut, expectedDimNum, return false);
  OP_CHECK_WRONG_DIMENSION(indicesOut, expectedDimNum, return false);
  return true;
}

// 检查参数情况
static aclnnStatus CheckParamsDim(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut,
                                  aclTensor *indicesOut)
{
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNullDim(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查参数的数据类型是否符合预期
  CHECK_RET(CheckDtypeValidDim(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查dim是否为 [-N, N-1] 范围内
  CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否超出MAX_DIM
  CHECK_RET(CheckShapeDim(self, dim, keepDim, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

  // 5. 当self的数据类型为BFLOAT16时，self.shape[dim]不能等于1
  auto dimSize = self->GetViewShape().GetDimNum();
  if (dimSize != 0) {
    auto realDim = (dim + dimSize) % dimSize;
    if (self->GetDataType() == op::DataType::DT_BF16 && self->GetViewShape().GetDim(realDim) == 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim axis value is not support 1 when input type is BF16.");
      return ACLNN_ERR_PARAM_INVALID;
    }
  }

  return ACLNN_SUCCESS;
}

// 将dim与最后一个dim进行对换
static aclIntArray* GetPermResult(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
  std::vector<int64_t> valuePerm(dimSize, 0);
  for (int64_t i = 0; i < dimSize; i++) {
    valuePerm[i] = i;
  }

  std::swap(valuePerm[dim], valuePerm[dimSize - 1]);

  return executor->AllocIntArray(valuePerm.data(), dimSize);
}

static const aclTensor* MedianAdaptInputZeroDimTensor(const aclTensor *self, int64_t dimNum, aclOpExecutor *executor) {
  if (dimNum != 0) {
    return self;
  }
  int64_t selfShapeValue[1] = {1};
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
  auto selfReshape = l0op::Reshape(self, selfShape, executor);
  return selfReshape;
}

static const aclTensor* ReduceOneDim(const aclTensor *self, int64_t selfShapeDim, aclOpExecutor *executor) {
  int64_t selfShapeValue[1] = {1};
  for (int64_t i = 0; i < selfShapeDim; i++) {
    selfShapeValue[0] *= self->GetViewShape().GetDim(i);
  }
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
  CHECK_RET(selfShape != nullptr, nullptr);
  auto selfReshape = l0op::Reshape(self, selfShape, executor);
  CHECK_RET(selfReshape != nullptr, nullptr);
  return selfReshape;
}

static aclIntArray* GetReduceShape(const aclTensor *self, const int64_t dim, aclOpExecutor *executor) {
  int64_t dimSize = GetTensorDim(self);
  int64_t selfShapeValue[dimSize - 1];
  int64_t idx = 0;
  for (int64_t i = 0; i < dimSize; i++) {
    if (i == dim) {
      continue;
    }
    selfShapeValue[idx] = self->GetViewShape().GetDim(i);
    idx++;
  }
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, dimSize - 1);
  CHECK_RET(selfShape != nullptr, nullptr);
  return selfShape;
}

static std::tuple<const aclTensor*, const aclTensor*> SortProcess(const aclTensor *self, const int64_t dim,
                                                                  aclOpExecutor *executor) {
  int64_t dimSize = GetTensorDim(self);
  CHECK_RET(dimSize != 0, std::tuple(nullptr, nullptr));
  int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
  bool descending = false;
  bool stable = true;

  const aclTensor* sortValues;
  const aclTensor* sortIndices;

  auto realDim = (dim + dimSize) % dimSize;
  if (realDim == dimMax) {
    auto sortResult = l0op::Sort(self, -1, descending, stable, op::DataType::DT_INT32, executor);
    CHECK_RET(CheckTupleNotNullptr(sortResult), std::tuple(nullptr, nullptr));
    sortValues = std::get<0>(sortResult);
    sortIndices = std::get<1>(sortResult);
  } else {
    // dim非最后一维，需进行transpose
    auto valuePerm = GetPermResult(realDim, dimSize, executor);
    CHECK_RET(valuePerm != nullptr, std::tuple(nullptr, nullptr));
    auto transposeSelf = l0op::Transpose(self, valuePerm, executor);
    CHECK_RET(transposeSelf != nullptr, std::tuple(nullptr, nullptr));

    auto sortResult = l0op::Sort(transposeSelf, -1, descending, stable, op::DataType::DT_INT32, executor);
    CHECK_RET(CheckTupleNotNullptr(sortResult), std::tuple(nullptr, nullptr));
    sortValues = l0op::Transpose(std::get<0>(sortResult), valuePerm, executor);
    sortIndices = l0op::Transpose(std::get<1>(sortResult), valuePerm, executor);
  }

  return std::tie(sortValues, sortIndices);
}

static const aclTensor* GetLast(const aclTensor *sortValues, int64_t dim, aclOpExecutor *executor) {
  int64_t LastIndex = sortValues->GetViewShape().GetDim(dim) - 1;
  const aclTensor *LastIndexTensor = executor->ConvertToTensor(&LastIndex, 1, op::DataType::DT_INT64);
  CHECK_RET(LastIndexTensor != nullptr, nullptr);
  auto lastResult = l0op::GatherV2(sortValues, dim, LastIndexTensor, executor);
  CHECK_RET(lastResult != nullptr, nullptr);
  return lastResult;
}

static const aclTensor* CreateNanTensor(const aclTensor *self, aclOpExecutor *executor) {
  FVector<float> valVector = {NAN};
  auto nanTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), self->GetDataType());
  return nanTensor;
}

static const aclTensor* GetNanMask(const aclTensor *self, int64_t dim, aclOpExecutor *executor) {
  // 判断是否为nan，nan的notEqual结果是true，非nan的notEqual结果是false
  auto lastValue = GetLast(self, dim, executor);
  CHECK_RET(lastValue != nullptr, nullptr);
  auto nanMask = l0op::NotEqual(lastValue, lastValue, executor);
  CHECK_RET(nanMask != nullptr, nullptr);
  return nanMask;
}

static const aclTensor *GetFirstNanIndices(const aclTensor *sortValues, const aclTensor *sortIndices, int64_t dim,
                                           aclOpExecutor *executor) {
  // nan的Equal结果是false，非nan的Equal结果是true
  auto equalTensor = l0op::Equal(sortValues, sortValues, executor);
  OP_CHECK(equalTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Equal return nullptr."), nullptr);

  auto endIndices= sortValues->GetViewShape().GetDim(dim) - 1;
  const aclTensor *endTensor = executor->ConvertToTensor(executor->AllocScalar(endIndices), sortIndices->GetDataType());
  OP_CHECK(endTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), nullptr);

  auto maskedFillTensor = l0op::MaskedFill(sortIndices, equalTensor, endTensor, executor);
  OP_CHECK(maskedFillTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedFill return nullptr."), nullptr);

  int64_t appendDim[1] = {dim};
  auto dimArray = executor->AllocIntArray(appendDim, 1);
  OP_CHECK(dimArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocIntArray return nullptr."), nullptr);
  auto reduceMinTensor = l0op::ReduceMin(maskedFillTensor, dimArray, true, executor);
  OP_CHECK(reduceMinTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceMin return nullptr."), nullptr);

  return reduceMinTensor;
}

static std::tuple<const aclTensor*, const aclTensor*> GetMedianSortResult(const aclTensor *self,
                                                                          int64_t dim,
                                                                          aclIntArray *selfReduceShape,
                                                                          aclOpExecutor *executor) {
  // 调用Sort算子
  auto sortResult = SortProcess(self, dim, executor);
  auto sortValues = std::get<0>(sortResult);
  CHECK_RET(sortValues != nullptr, std::tuple(nullptr, nullptr));
  auto sortIndices = std::get<1>(sortResult);
  CHECK_RET(sortIndices != nullptr, std::tuple(nullptr, nullptr));

  // GatherV2选择中位数
  auto medianIndex = (self->GetViewShape().GetDim(dim) - 1) / 2;
  const aclTensor *medianIndexTensor = executor->ConvertToTensor(&medianIndex, 1, op::DataType::DT_INT64);
  CHECK_RET(medianIndexTensor != nullptr, std::tuple(nullptr, nullptr));
  auto valueResult = l0op::GatherV2(sortValues, dim, medianIndexTensor, executor);
  CHECK_RET(valueResult != nullptr, std::tuple(nullptr, nullptr));
  auto indicesResult = l0op::GatherV2(sortIndices, dim, medianIndexTensor, executor);
  CHECK_RET(indicesResult != nullptr, std::tuple(nullptr, nullptr));

  auto lastValueResult = valueResult;
  auto lastIndicesResult = indicesResult;
  auto FLOAT_DTYPE_LIST = GetFloatList();
  if (CheckType(self->GetDataType(), FLOAT_DTYPE_LIST)) {
    // 获取第一个nan的索引
    auto indicesResultTmp = GetFirstNanIndices(sortValues, sortIndices, dim, executor);
    CHECK_RET(indicesResultTmp != nullptr, std::tuple(nullptr, nullptr));

    auto nanMask = GetNanMask(sortValues, dim, executor);
    CHECK_RET(nanMask != nullptr, std::tuple(nullptr, nullptr));

    // 生成nan的Tensor
    auto nanTensor = CreateNanTensor(valueResult, executor);
    CHECK_RET(nanTensor != nullptr, std::tuple(nullptr, nullptr));

    int64_t dimSize = GetTensorDim(self);
    int64_t selfShapeValue[dimSize];
    int64_t idx = 0;
    for (int64_t i = 0; i < dimSize; i++) {
      if (i == dim) {
        selfShapeValue[idx] = 1;
        idx++;
        continue;
      }
      selfShapeValue[idx] = self->GetViewShape().GetDim(i);
      idx++;
    }
    aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, dimSize);
    CHECK_RET(selfShape != nullptr, std::tuple(nullptr, nullptr));

    auto selfShapeTensor = executor->ConvertToTensor(selfShape, op::ToOpDataType(ACL_INT64));
    CHECK_RET(selfShapeTensor != nullptr, std::tuple(nullptr, nullptr));
    auto nanFillTensor = l0op::Fill(selfShapeTensor, nanTensor, selfShape, executor);
    CHECK_RET(nanFillTensor != nullptr, std::tuple(nullptr, nullptr));

    lastValueResult = l0op::SelectV2(nanMask, nanFillTensor, valueResult, executor);
    lastIndicesResult = l0op::SelectV2(nanMask, indicesResultTmp, indicesResult, executor);
  }
  CHECK_RET(lastValueResult != nullptr, std::tuple(nullptr, nullptr));
  CHECK_RET(lastIndicesResult != nullptr, std::tuple(nullptr, nullptr));

  return std::tie(lastValueResult, lastIndicesResult);
}

aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor *self, aclTensor *valuesOut, uint64_t *workspaceSize,
                                        aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMedian, DFX_IN(self), DFX_OUT(valuesOut));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, valuesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty() || valuesOut->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfShapeDim = GetTensorDim(self);
  auto selfReshape = selfContiguous;
  // 对不为1维的tensor进行reshape
  if (selfShapeDim != 1) {
    selfReshape = ReduceOneDim(selfContiguous, selfShapeDim, uniqueExecutor.get());
    CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  auto sortValues = selfReshape;
  // 调用l0算子Sort进行计算
  if (selfReshape->GetViewShape().GetDim(DIM_ZERO) > 1) {
    bool descending = false;
    bool stable = true;
    auto sortResult = l0op::Sort(selfReshape, -1, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(CheckTupleNotNullptr(sortResult), ACLNN_ERR_INNER_NULLPTR);
    sortValues = std::get<0>(sortResult);
  }
  CHECK_RET(sortValues != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto medianIndex = (sortValues->GetViewShape().GetDim(DIM_ZERO) - 1) / 2;
  const aclTensor *medianIndexTensor = uniqueExecutor->ConvertToTensor(&medianIndex, 1, op::DataType::DT_INT64);
  CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto selectResult = l0op::GatherV2(sortValues, 0, medianIndexTensor, uniqueExecutor.get());
  CHECK_RET(selectResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto result = selectResult;
  auto FLOAT_DTYPE_LIST = GetFloatList();
  if (CheckType(self->GetDataType(), FLOAT_DTYPE_LIST)) {
    auto nanMask = GetNanMask(sortValues, 0, uniqueExecutor.get());
    CHECK_RET(nanMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 生成nan的Tensor
    auto nanTensor = CreateNanTensor(selectResult, uniqueExecutor.get());
    CHECK_RET(nanTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    result = l0op::SelectV2(nanMask, nanTensor, selectResult, uniqueExecutor.get());
  }
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output shape check
  CHECK_RET(CheckReduceOutShape(result, valuesOut), ACLNN_ERR_PARAM_INVALID);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(result, valuesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMedian(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMedian);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim,
                                           aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnMedianDim, DFX_IN(self, dim, keepDim), DFX_OUT(valuesOut, indicesOut));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParamsDim(self, dim, keepDim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty() || valuesOut->IsEmpty() || indicesOut->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  int64_t dimSize = GetTensorDim(self);
  int64_t realDim = self->GetViewShape().IsScalar() ? 0 : dim < 0 ? dim + dimSize : dim;
  auto selfReshape = MedianAdaptInputZeroDimTensor(selfContiguous, dimSize, uniqueExecutor.get());
  CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  aclIntArray *selfReduceShape = GetReduceShape(selfReshape, realDim, uniqueExecutor.get());
  CHECK_RET(selfReduceShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t sortDimSize = self->GetViewShape().GetDim(static_cast<size_t>(realDim));
  const aclTensor *lastValueResult = nullptr;
  const aclTensor *lastIndicesResult = nullptr;
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && dimSize > 1 &&
      self->GetDataType() == op::DataType::DT_FLOAT && sortDimSize <= SMALLSORTLIMIT) {
    // small rank dim
    // Transpose
    std::vector<int64_t> perm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        perm[i] = i;
    }
    std::swap(perm[realDim], perm[0]);
    auto valuePerm = uniqueExecutor->AllocIntArray(perm.data(), dimSize);
    CHECK_RET(valuePerm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (realDim != 0) {
        selfReshape = l0op::Transpose(selfReshape, valuePerm, uniqueExecutor.get());
        CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // MedianDim
    auto result = l0op::MedianDim(selfReshape, 0, true, uniqueExecutor.get());
    lastValueResult = std::get<0>(result);
    CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    lastIndicesResult = std::get<1>(result);
    CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // Transpose back
    if (realDim != 0) {
        lastValueResult = l0op::Transpose(lastValueResult, valuePerm, uniqueExecutor.get());
        CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        lastIndicesResult = l0op::Transpose(lastIndicesResult, valuePerm, uniqueExecutor.get());
        CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
  } else {
    // 获取排序结果
    auto result = GetMedianSortResult(selfReshape, realDim, selfReduceShape, uniqueExecutor.get());
    lastValueResult = std::get<0>(result);
    CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    lastIndicesResult = std::get<1>(result);
    CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // Reshape
  auto valueReshape = keepDim ? lastValueResult : l0op::Reshape(lastValueResult, selfReduceShape, uniqueExecutor.get());
  CHECK_RET(valueReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto indicesReshape = keepDim ? lastIndicesResult : l0op::Reshape(lastIndicesResult, selfReduceShape,
                                                                    uniqueExecutor.get());
  CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Cast转换
  auto indicesCastResult = l0op::Cast(indicesReshape, indicesOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(indicesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // output shape check
  CHECK_RET(CheckReduceOutShape(valueReshape, valuesOut), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckReduceOutShape(indicesCastResult, indicesOut), ACLNN_ERR_PARAM_INVALID);
  // 如果出参valuesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto valueCopyResult = l0op::ViewCopy(valueReshape, valuesOut, uniqueExecutor.get());
  CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 如果出参indicesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto indicesCopyResult = l0op::ViewCopy(indicesCastResult, indicesOut, uniqueExecutor.get());
  CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMedianDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMedianDim);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus DealNanMedianEmptyTensor(aclTensor *out, aclOpExecutor *executor) {
  auto outShape = out->GetViewShape();
  op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(outShape);
  auto shapes = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
  const aclTensor *dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT64);

  const aclScalar *valueScalar = executor->AllocScalar(NAN);
  const aclTensor *valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
  auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
  CHECK_RET(fillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto dstCopyResult = l0op::ViewCopy(fillTensor, out, executor);
  CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static int64_t GetTensorSize(const aclTensor *self) {
  int64_t size = 1;
  if (self->IsEmpty()) {
    return size;
  }
  int64_t dimNum = self->GetViewShape().GetDimNum();
  int64_t dim_post_expr = self->GetViewShape().GetDimNum();
  for (int64_t dim = 0; dim < dimNum; dim++) {
    size *= self->GetViewShape().GetDim(dim);
  }

  return size;
}

// 获取带dim的median中位数索引
static const aclTensor *GetNanMedianDimIndexTensor(const aclTensor *selfReshape, int64_t dim, aclOpExecutor *executor) {
  // nan的Equal结果是false，非nan的Equal结果是true
  auto selfEqualTensor = l0op::Equal(selfReshape, selfReshape, executor);
  OP_CHECK(selfEqualTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Equal return nullptr."), nullptr);

  auto dimSize = selfReshape->GetStorageShape().GetDim(dim);

  // 将Equal的结果cast成int类型
  const aclTensor *equalCasted = nullptr;
  auto dtype = op::DataType::DT_INT32;
  auto size = GetTensorSize(selfEqualTensor);
  if (size > MAX_INT32) {
    equalCasted = l0op::Cast(selfEqualTensor, op::DataType::DT_INT64, executor);
    dtype = op::DataType::DT_INT64;
  } else {
    equalCasted = l0op::Cast(selfEqualTensor, op::DataType::DT_INT32, executor);
  }
  OP_CHECK(equalCasted != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast return nullptr."), nullptr);

  // 通过l0算子ReduceSumOp求非nan元素的个数
  const int64_t reduceDim[] = {dim};
  auto reduceDimArray = executor->AllocIntArray(reduceDim, 1);
  auto reduceSumTensor = l0op::ReduceSumOp(equalCasted, reduceDimArray, true, executor);
  OP_CHECK(reduceSumTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "reducesumop return nullptr."), nullptr);

  // 减1
  auto one = 1;
  const aclTensor *oneTensor = executor->ConvertToTensor(&one, 1, dtype);
  OP_CHECK(oneTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), nullptr);

  auto subOneTensor = l0op::Sub(reduceSumTensor, oneTensor, executor);
  OP_CHECK(subOneTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Sub return nullptr."), nullptr);
  // 调用l0算子将减1的结果除以2
  const aclTensor* resultTensor = nullptr;
  if (dimSize >= MAX_CONVERT_NUM) {
    int64_t rightShiftisor = 1;
    const aclTensor *otherTensor = executor->ConvertToTensor(&rightShiftisor, 1, dtype);
    OP_CHECK(otherTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), nullptr);

    // 调用l0算子rightShift计算中位数索引
    resultTensor = l0op::RightShift(subOneTensor, otherTensor, executor);
    OP_CHECK(resultTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "rightShift return nullptr."), nullptr);
  } else {
    int64_t Divisor = 2;
    const aclTensor *otherTensor = executor->ConvertToTensor(&Divisor, 1, dtype);
    OP_CHECK(otherTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), nullptr);

    // 调用l0算子rightShift计算中位数索引
    resultTensor = l0op::Div(subOneTensor, otherTensor, executor);
    OP_CHECK(resultTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Div return nullptr."), nullptr);
  }

  return resultTensor;
}

aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnNanMedian, DFX_IN(self), DFX_OUT(out));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    DealNanMedianEmptyTensor(out, uniqueExecutor.get());
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t selfShapeSize = 0;
  auto selfShapeDim = GetTensorDim(self);
  auto selfReshape = selfContiguous;

  // 对不为1维的tensor进行reshape
  if (selfShapeDim > 1) {
    int64_t selfShapeValue[1] = {1};
    for (int64_t i = 0; i < selfShapeDim; i++) {
      selfShapeValue[0] *= self->GetViewShape().GetDim(i);
    }

    aclIntArray *selfShape = uniqueExecutor->AllocIntArray(selfShapeValue, 1);
    CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfReshape = l0op::Reshape(selfContiguous, selfShape, uniqueExecutor.get());
    CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 调用l0算子Sort进行计算
  bool descending = false;
  bool stable = true;
  auto sortResult = l0op::Sort(selfReshape, -1, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
  CHECK_RET(CheckTupleNotNullptr(sortResult), ACLNN_ERR_INNER_NULLPTR);
  auto sortValues = std::get<0>(sortResult);

  // 获取中位数的索引
  auto medianIndexTensor = GetNanMedianDimIndexTensor(selfReshape, 0, uniqueExecutor.get());
  CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 按照索引取出中位数
  auto selectResult = l0op::GatherV2(sortValues, 0, medianIndexTensor, uniqueExecutor.get(), 0, true);
  CHECK_RET(selectResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output shape check
  CHECK_RET(CheckReduceOutShape(selectResult, out), ACLNN_ERR_PARAM_INVALID);
  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(selectResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedian(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnNanMedian);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus ExecZeroDimNum(const aclTensor *self, int64_t dim, aclTensor *valuesOut,
                                  aclTensor *indicesOut, aclOpExecutor *executor) {
  CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

  auto valueCopyResult = l0op::ViewCopy(self, valuesOut, executor);
  CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 标量的索引直接返回0
  auto indicesOutZero = l0op::ZerosLike(indicesOut, executor);
  CHECK_RET(indicesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesCopyResult = l0op::ViewCopy(indicesOutZero, indicesOut, executor);
  CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim,
                                           aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnNanMedianDim, DFX_IN(self, dim, keepDim), DFX_OUT(valuesOut, indicesOut));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParamsDim(self, dim, keepDim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

   // 如果传入是标量
  auto dimNum = self->GetViewShape().GetDimNum();
  if (dimNum == 0) {
    ret = ExecZeroDimNum(self, dim, valuesOut, indicesOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  dim = (dim + dimNum) % dimNum;

  // 处理空tensor
  if (self->IsEmpty()) {
    if (self->GetViewShape().GetDim(dim) == 0) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction dim %ld to have non-zero size.", dim);
      CHECK_RET(false, ACLNN_ERR_PARAM_INVALID);
    }

    // 多维空tensor返回空tensor
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

   // 获取中位数的索引
  auto medianIndexTensor = GetNanMedianDimIndexTensor(selfContiguous, dim, uniqueExecutor.get());
  CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Sort算子
  auto sortResult = SortProcess(selfContiguous, dim, uniqueExecutor.get());
  auto sortValues = std::get<0>(sortResult);
  CHECK_RET(sortValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto sortIndices = std::get<1>(sortResult);
  CHECK_RET(sortIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto valueResult = l0op::GatherElements(sortValues, dim, medianIndexTensor, uniqueExecutor.get());
  CHECK_RET(valueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto indicesResult = l0op::GatherElements(sortIndices, dim, medianIndexTensor, uniqueExecutor.get());
  CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Reshape
  aclIntArray *selfShape = GetReduceShape(self, dim, uniqueExecutor.get());
  CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto ValueReshape = keepDim ? valueResult : l0op::Reshape(valueResult, selfShape, uniqueExecutor.get());
  CHECK_RET(ValueReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto IndicesReshape = keepDim ? indicesResult : l0op::Reshape(indicesResult, selfShape, uniqueExecutor.get());
  CHECK_RET(IndicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Cast转换
  auto indicesCastResult = l0op::Cast(IndicesReshape, indicesOut->GetDataType(), uniqueExecutor.get());
  CHECK_RET(indicesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output shape check
  CHECK_RET(CheckReduceOutShape(ValueReshape, valuesOut), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckReduceOutShape(indicesCastResult, indicesOut), ACLNN_ERR_PARAM_INVALID);
  // 如果出参valuesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto valueCopyResult = l0op::ViewCopy(ValueReshape, valuesOut, uniqueExecutor.get());
  CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 如果出参indicesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto indicesCopyResult = l0op::ViewCopy(indicesCastResult, indicesOut, uniqueExecutor.get());
  CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedianDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnNanMedianDim);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif