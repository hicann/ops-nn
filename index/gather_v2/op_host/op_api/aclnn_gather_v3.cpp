/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_gather_v3.h"
#include "gather_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
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
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static const int64_t DIM0 = 0;
static const int64_t MAX_DIM_LEN = 8;
static const int64_t HIGH_PRECISION = 0;
static const int64_t HIGH_PERFORMANCE = 1;
static const int64_t SUPPORT_OUT_OF_BOUND_INDEX = 2;
static const int64_t MIN_SELF_SIZE = 61440;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_INT16,
  DataType::DT_INT8, DataType::DT_UINT8, DataType::DT_BOOL, DataType::DT_DOUBLE, DataType::DT_COMPLEX64,
  DataType::DT_COMPLEX128, DataType::DT_BF16};

static const std::initializer_list<DataType> INDEX_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static inline bool HasEmptyTensor(const aclTensor *self, const aclTensor *index) {
  return self->IsEmpty() || index->IsEmpty();
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // self、index、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_CURRENT = DTYPE_SUPPORT_LIST;

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_CURRENT, return false);

  // 检查index的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_SUPPORT_LIST, return false);

  return true;
}

static inline bool CheckOutShape(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *out) {
  // 根据算子语义，推导算子输出shape
  Shape outShape;
  if (self->GetViewShape().GetDimNum() == 0) {
    outShape = self->GetViewShape();
  } else {
    for (int64_t i = 0; i < dim; i++) {
      outShape.AppendDim(self->GetViewShape().GetDim(i));
    }
    for (size_t i = 0; i < index->GetViewShape().GetDimNum(); i++) {
      outShape.AppendDim(index->GetViewShape().GetDim(i));
    }

    for (size_t i = dim + 1; i < self->GetViewShape().GetDimNum(); i++) {
      outShape.AppendDim(self->GetViewShape().GetDim(i));
    }
  }

  // 判断out的shape与推导出的输出shape是否相等
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);

  return true;
}

static inline bool CheckShape(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *out) {
  // self、index的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(index, MAX_DIM_LEN, return false);

  int64_t selfDimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (selfDimSize == 0) {
    selfDimSize = 1;
  }

  // dim必须在self的维度范围内
  if (!(dim >= -selfDimSize && dim < selfDimSize)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim %ld not in valid range: [%ld, %ld)", dim, -selfDimSize, selfDimSize);
    return false;
  } else if (dim < 0) {
    dim += selfDimSize;
  }

  return CheckOutShape(self, dim, index, out);
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, const aclTensor *index, int64_t batchDims,
                                int64_t& mode, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, dim, index, out), ACLNN_ERR_PARAM_INVALID);

  if (dim != 0 || batchDims !=0 || mode != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Currently, only dim=0, batchDims=0, and mode=1 are supported.");
    return ACLNN_ERR_PARAM_INVALID;
  }

  auto xsize = ge::GetSizeByDataType(self->GetDataType());
  auto selfDimNum = self->GetViewShape().GetDimNum();
  for (uint64_t i = 0; i < selfDimNum ; i++) {
    xsize *= self->GetViewShape().GetDim(i);
  }

  if (xsize < MIN_SELF_SIZE) {
    mode = HIGH_PRECISION;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index, int64_t batchDims,
                                          int64_t mode, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnGatherV3, DFX_IN(self, index), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, index, batchDims, mode, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 算子的空tensor的支持情况
  if (HasEmptyTensor(self, index)) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // index如果非连续，需要转连续
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indexParam = indexContiguous;
  // 判断index是零维的情况：将它转成一维
  if (indexContiguous->GetViewShape().GetDimNum() == 0) {
    const int64_t zero = 0;
    indexParam = l0op::UnsqueezeNd(indexContiguous, zero, uniqueExecutor.get());
    CHECK_RET(indexParam != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 调用l0算子GatherV2进行计算
  auto gatherV2Result = l0op::GatherV2WithImplMode(selfContiguous, dim, indexContiguous, mode,
                                                    uniqueExecutor.get(), batchDims);
  CHECK_RET(gatherV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(gatherV2Result, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGatherV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGatherV3);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
