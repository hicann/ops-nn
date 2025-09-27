/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_flip.cpp
 * \brief
 */

#include "aclnn_flip.h"
#include "reverse_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,
    op::DataType::DT_BOOL, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BF16,
    op::DataType::DT_BOOL, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<DataType> ASCEND910_95_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_UINT16,
  op::DataType::DT_INT32, op::DataType::DT_UINT32, op::DataType::DT_INT64, op::DataType::DT_UINT64,
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    return ASCEND910_95_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static inline bool CheckDuplicate(int64_t dim, FVector<int64_t>& vecDims) {
  for (int64_t d : vecDims) {
    if (d == dim) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim:[%ld] appears multiple times in the list of dims", dim);
      return true;
    }
  }

  vecDims.push_back(dim);
  return false;
}

static bool CheckDims(const aclIntArray *dims, const aclTensor *self) {
  // 获取输入tensor的维数，并得出轴的最大值和最小值
  int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if (dimNum == 0) {
    dimNum = 1; // this will make range [-1, 0]
  }
  const int64_t min = -dimNum;
  const int64_t max = dimNum - 1;

  OP_LOGD("CheckDims dimNum:%ld, min:%ld, max:%ld", dimNum, min, max);

  const int64_t *dimsData = dims->GetData();
  FVector<int64_t> dimVec;
  for (uint64_t i = 0; i < dims->Size(); ++i) {
    int64_t dim = dimsData[i];

    // dims中的value不在有效范围时，则报错
    if (dim < min || dim > max) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
              min, max, dim);
      return false;
    }

    // 将负数轴做转换为非负数轴，并检查是否存在重复
    dim = (dim < 0) ? (dim + dimNum) : dim;
    if (CheckDuplicate(dim, dimVec)) {
      return false;
    }
  }

  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  return true;
}

static inline bool CheckNotNull(const aclTensor *self, const aclIntArray *dims, const aclTensor *out) {
  // self、dims、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(dims, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  // 检查数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *dims, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, dims, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查dim的取值
  CHECK_RET(CheckDims(dims, self), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool HasEmptyTensor(const aclTensor *self) {
  if (self->IsEmpty()) {
    return true;
  }

  return false;
}

aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnFlip, DFX_IN(self, dims), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dims, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (HasEmptyTensor(self)) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子ReverseV2进行计算
  auto reverseV2Result = l0op::ReverseV2(selfContiguous, dims, uniqueExecutor.get());
  CHECK_RET(reverseV2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(reverseV2Result, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnFlip);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}