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
 * \file aclnn_index_fill_tensor.cpp
 * \brief
 */

#include "aclnn_index_fill_tensor.h"
#include "index_fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/framework_op.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM = 8;

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> DTYPE_910B_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclIntArray *index,
                         const aclScalar *value, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(value, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *index, int64_t dim, const aclTensor *out) {
  if (self->IsEmpty()) {
    return true;
  }
  // 校验self的shape是否等于out的shape
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  // 最大维度限制
  OP_CHECK_MAX_DIM(self, MAX_DIM, return false);

  auto selfShape = self->GetViewShape();
  auto selfDim = static_cast<int64_t>(selfShape.GetDimNum());
  if ((dim != 0 && dim >= selfDim) || (dim == 0 && dim > selfDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim[%ld] is greater than self dim[%ld].", dim, selfDim);
    return false;
  }

  if ((dim < 0 && (dim * (-1)) > selfDim && selfDim > 0) || (dim < 0 && (dim * (-1)) > 1 && selfDim == 0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, abs(input dim[%ld]) is greater than self dim[%ld].",
            dim, selfDim);
    return false;
  }

  int64_t transferDim = dim >= 0 ? dim : (selfDim > 0 ? (dim + selfDim) : 0);
  for (int64_t i = 0; i < static_cast<int64_t>(index->Size()); i++) {
    auto dimSize = static_cast<int64_t>(selfShape.GetDim(transferDim));
    if ((*index)[i] >= dimSize || (*index)[i] < (-dimSize)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index value[%ld] is out of range, it should be smaller than [%ld].",
             (*index)[i], selfShape.GetDim(transferDim));
      return false;
    }
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在算子的支持列表内
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B ||
      socVersion == SocVersion::ASCEND910_93 || socVersion == SocVersion::ASCEND910_95) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_910B_SUPPORT_LIST, return false);
  } else {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  }
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
  return true;
}

static bool CheckPromoteType(const aclScalar *value) {
  if (IsComplexType(value->GetDataType())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Type of value do not support complex type");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, const aclIntArray *index,
                               const aclScalar *value, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, value, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的shape是否满足要求
  CHECK_RET(CheckShape(self, index, dim, out), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查value的类型
  CHECK_RET(CheckPromoteType(value), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclIntArray *GenerateShapeArray(const aclTensor *self, aclOpExecutor *executor)
{
  auto selfShape = self->GetViewShape();
  aclIntArray *shapeArray;
  FVector<int64_t> tmp;
  int64_t selfDim = selfShape.GetDimNum();
  for (int64_t idx = 0; idx < selfDim; idx++) {
    int64_t tmpVal = selfShape.GetDim(idx);
    tmp.push_back(tmpVal);
  }
  shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());
  return shapeArray;
}

static const aclTensor *ReshapeTensor(const aclTensor *self, const aclIntArray *shapeArray, aclOpExecutor *executor)
{
  const aclTensor *reshapedTensor = l0op::Reshape(self, shapeArray, executor);
  return reshapedTensor;
}

static const aclTensor *GenerateAssistMatrix(const aclTensor *self, const aclIntArray *index, int64_t dim, bool flag,
    float value, op::DataType type, aclOpExecutor *executor)
{
  int blocksize = 0;
  int blocknum = 1;
  int n = 1;
  aclIntArray *shapeArray = GenerateShapeArray(self, executor);
  CHECK_RET(shapeArray != nullptr, nullptr);
  auto selfShape = self->GetViewShape();
  int64_t selfDim = static_cast<int64_t>(selfShape.GetDimNum());
  for (int64_t i = 0; i < selfDim; i++) {
    if (i <= dim) {
      blocknum *= selfShape[i];
    }
    n *= selfShape[i];
  }
  blocksize = n / blocknum;

  float initVal = flag ? 1 : 0;
  auto assist = executor->AllocHostTensor({n}, op::DataType::DT_FLOAT);
  CHECK_RET(assist != nullptr, nullptr);
  float *addr = static_cast<float*>(assist->GetStorageAddr());
  for (int i = 0; i < n; i++) {
    assist->SetData(i, initVal, op::DataType::DT_FLOAT);
  }

  for (uint64_t i = 0; i < index->Size(); i++) {
    uint64_t start = 0, end = 0;
    uint64_t idx = (*index)[i];
    uint64_t k = idx, count = 0;
    while (k < static_cast<uint64_t>(blocknum)) {
      start = blocksize * k;
      end = start + blocksize;
      for (uint64_t j = start; j < end; j++) {
        addr[j] = value;
      }
      count++;
      k = idx + selfShape[dim] * count;
    }
  }

  auto deviceTensor = op::CopyToNpu(assist, executor);
  CHECK_RET(deviceTensor != nullptr, nullptr);
  auto castedTensor = l0op::Cast(deviceTensor, type, executor);
  CHECK_RET(castedTensor != nullptr, nullptr);
  return ReshapeTensor(castedTensor, shapeArray, executor);
}

aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclIntArray *index,
                                                 const aclScalar *value, aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnIndexFillTensor, DFX_IN(self, dim, index, value), DFX_OUT(out));
  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, index, value, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (index->Size() == 0) {
    auto viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  float fillVal = value->ToFloat();
  auto selfDim = self->GetViewShape().GetDimNum();
  dim = dim >= 0 ? dim : (selfDim > 0 ? (dim + selfDim) : 0);
  size_t indexNum = index->Size();
  int64_t appendIndex[indexNum];
  for (uint64_t i = 0; i < indexNum; i++) {
    if ((*index)[i] < 0) {
      appendIndex[i] = (*index)[i] + self->GetViewShape().GetDim(dim);
    } else {
      appendIndex[i] = (*index)[i];
    }
  }
  index = uniqueExecutor.get()->AllocIntArray(appendIndex, indexNum);

  const aclTensor *assist1 = GenerateAssistMatrix(self, index, dim, true, 0, self->GetDataType(),
                                                  uniqueExecutor.get());
  CHECK_RET(assist1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor *assist2 = GenerateAssistMatrix(self, index, dim, false, fillVal, self->GetDataType(),
                                                  uniqueExecutor.get());
  CHECK_RET(assist2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行IndexFill计算
  auto indexFillOut = l0op::IndexFillD(selfContiguous, assist1, assist2, dim, uniqueExecutor.get());
  CHECK_RET(indexFillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(indexFillOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnIndexFillTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceIndexFillTensorGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclIntArray *index,
                                                        const aclScalar *value, uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {
  return aclnnIndexFillTensorGetWorkspaceSize(selfRef, dim, index, value, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceIndexFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceIndexFillTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif