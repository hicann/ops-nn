/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kthvalue.h"
#include "level0/sort.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "gather_v2.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/cast.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910A_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BF16};

// 910D Sort的self/value支持的dtype
static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16, op::DataType::DT_UINT8,
  op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  }else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
    return ASCEND950_DTYPE_SUPPORT_LIST;
  }
  return ASCEND910A_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(values, return false);
  OP_CHECK_NULL(indices, return false);
  return true;
}

int64_t DimMakeWrap(int64_t dim, int64_t dimPostExpr) {
  // 支持0维tensor
  if (dimPostExpr <= 0) {
    dimPostExpr = 1;
  }
  if (dim < 0) {
    dim += dimPostExpr;
  }
  return dim;
}

static bool CheckParamValid(const aclTensor *self, int64_t k, int64_t dim) {
  // 检查参数dim是否合法
  auto inputShape = self->GetViewShape();
  int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
  if (tmpDim == 0 && dim != 0 && dim != -1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of input is invalid");
    return false;
  } else if (tmpDim > 0 && (dim < -tmpDim || dim >= tmpDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-%ld, %ld], but got %ld)",
            tmpDim, tmpDim - 1, dim);
    return false;
  }

  // 检查参数k是否合法
  dim = DimMakeWrap(dim, tmpDim);
  int64_t tmpK = (tmpDim > 0) ? inputShape.GetDim(dim) : 1;
  if (k <= 0 || k > tmpK) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Selected index k out or range (max num of self.size(%ld) is %ld, but k is %ld)",
            tmpDim, tmpK, k);
    return false;
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查values数据类型与self是否一致
  OP_CHECK_DTYPE_NOT_MATCH(values, self->GetDataType(), return false);

  // 检查indices是否为Long类型
  OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT64, return false);

  return true;
}

static bool CheckShape(const aclTensor *self) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t k, int64_t dim,
                               const aclTensor *values, const aclTensor *indices) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, values, indices), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查参数k和dim是否合法
  CHECK_RET(CheckParamValid(self, k, dim), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self、values和indices的数据类型是否合法
  CHECK_RET(CheckDtypeValid(self, values, indices), ACLNN_ERR_PARAM_INVALID);

  // 4. 查输入tensor的shape是否为异常
  CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor *SortAdaptInputZeroDimTensor(const aclTensor *self, int64_t dimNum, aclOpExecutor *executor) {
  if (dimNum != 0) {
    return self;
  }
  int64_t selfShapeValue[1] = {1};
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
  if (selfShape == nullptr) {
    return nullptr;
  }
  auto selfReshape = l0op::Reshape(self, selfShape, executor);
  return selfReshape;
}

static aclIntArray *GetDimTransposeArray(int64_t dimNum, int64_t positiveDim,
                                         int64_t lastDim, aclOpExecutor *executor) {
  std::vector<int64_t> perm(dimNum);
  for (int64_t i = 0; i < dimNum; i++) {
    perm[i] = i;
  }
  std::swap(perm[positiveDim], perm[lastDim]);
  return executor->AllocIntArray(perm.data(), dimNum);
}

const aclTensor *KthvalueCalculate(const aclTensor *x,
                                   int64_t k, int64_t dim, aclOpExecutor *executor) {
  FVector<int64_t> indexVector = {k - 1};
  int64_t dimNum = static_cast<int64_t>(x->GetViewShape().GetDimNum());
  int64_t positiveDim = DimMakeWrap(dim, dimNum);
  auto indexTensor = executor->ConvertToTensor(indexVector.data(), indexVector.size(), DataType::DT_INT64);
  return l0op::GatherV2(x, positiveDim, indexTensor, executor);
}

aclnnStatus aclnnKthvalueGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim,
                                          bool keepdim, aclTensor *valuesOut,
                                          aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnKthvalue, DFX_IN(self, k, dim, keepdim), DFX_OUT(valuesOut, indicesOut));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, k, dim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 支持空tensor
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t dimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
  int64_t positiveDim = DimMakeWrap(dim, dimNum);
  int64_t lastDim = DimMakeWrap(-1, dimNum);

  auto selfReshape = SortAdaptInputZeroDimTensor(selfContiguous, dimNum, uniqueExecutor.get());
  CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  bool descending = false;
  bool stable = true;

  const aclTensor *indicesSortOut = nullptr;
  const aclTensor *valuesSortOut = nullptr;

  if (positiveDim != lastDim) {
    aclIntArray *axes = GetDimTransposeArray(dimNum, positiveDim, lastDim, uniqueExecutor.get());
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对self进行transpose
    auto selfTranspose = l0op::Transpose(selfReshape, axes, uniqueExecutor.get());
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行sort计算
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
      lastDim = -1;
    }
    auto sortOut = l0op::Sort(selfTranspose, lastDim, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
    auto valuesSortOutNoTrans = std::get<0>(sortOut);
    auto indicesSortOutNoTrans = std::get<1>(sortOut);
    CHECK_RET(valuesSortOutNoTrans != nullptr && indicesSortOutNoTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果values进行transpose，转换成正确的shape
    valuesSortOut = l0op::Transpose(valuesSortOutNoTrans, axes, uniqueExecutor.get());
    CHECK_RET(valuesSortOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果indices进行transpose，转换成正确的shape
    indicesSortOut = const_cast<aclTensor *>(l0op::Transpose(indicesSortOutNoTrans, axes, uniqueExecutor.get()));
    CHECK_RET(indicesSortOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
      positiveDim = -1;
    }
    auto sortOut = l0op::Sort(selfReshape, positiveDim, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
    valuesSortOut = std::get<0>(sortOut);
    indicesSortOut = std::get<1>(sortOut);
    CHECK_RET(valuesSortOut != nullptr && indicesSortOut!= nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 查找第k个最小值及下标
  auto values = KthvalueCalculate(valuesSortOut, k, dim, uniqueExecutor.get());
  auto indices = KthvalueCalculate(indicesSortOut, k, dim, uniqueExecutor.get());
  CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *valuesSqueeze = nullptr;
  const aclTensor *indicesSqueeze = nullptr;
  if (!keepdim) {
    // 将dim压缩
    valuesSqueeze = l0op::SqueezeNd(values, dim, uniqueExecutor.get());
    indicesSqueeze = l0op::SqueezeNd(indices, dim, uniqueExecutor.get());
    CHECK_RET(valuesSqueeze != nullptr && indicesSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  if (keepdim) {
    CHECK_RET(CheckReduceOutShape(values, valuesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indices, indicesOut), ACLNN_ERR_PARAM_INVALID);
  } else {
    CHECK_RET(CheckReduceOutShape(valuesSqueeze, valuesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesSqueeze, indicesOut), ACLNN_ERR_PARAM_INVALID);
  }

  // 将values结果拷贝到valuesOut上
  auto viewCopyValuesResult = l0op::ViewCopy((keepdim ? values : valuesSqueeze), valuesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyValuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将结果indices进行cast，转换成int64类型
  auto indicesCastInt64 = l0op::Cast(keepdim ? indices : indicesSqueeze, op::DataType::DT_INT64, uniqueExecutor.get());
  CHECK_RET(indicesCastInt64 != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将indices_cast_int64结果拷贝到indicesOut上
  auto viewCopyIndicesResult = l0op::ViewCopy(indicesCastInt64, indicesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnKthvalue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnKthvalue);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
