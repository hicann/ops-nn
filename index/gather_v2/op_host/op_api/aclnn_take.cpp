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
 * \file aclnn_take.cpp
 * \brief
 */
#include "aclnn_take.h"
#include "gather_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr uint64_t MAX_INPUT_DIM_NUM = 8;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16,
  DataType::DT_UINT64, DataType::DT_INT64, DataType::DT_UINT32, DataType::DT_INT32,
  DataType::DT_UINT16, DataType::DT_INT16, DataType::DT_UINT8,  DataType::DT_INT8,
  DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BOOL,
  DataType::DT_BF16};

static const std::initializer_list<DataType> DTYPE_INDEX_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 检查index的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(index, DTYPE_INDEX_LIST, return false);

  // 仅910B支持BF16
  if (self->GetDataType() == DataType::DT_BF16) {
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "take operator doesn't support [%s], except on ASCEND910B or ASCEND910_95 chip",
              op::ToString(self->GetDataType()).GetString());
      return false;
    }
  }

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // index 维度有上限值
  OP_CHECK_MAX_DIM(index,  MAX_INPUT_DIM_NUM, return false);

  // out和index的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, index, return false);

  if (self->IsEmpty() && !index->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IndexError: take(): tried to take from an empty tensor");
    return false;
  }
  return true;
}

static inline bool CheckFormat(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(self->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, self [%s]",
            ToString(self->GetStorageFormat()).GetString());
    return false;
  }
  if (op::IsPrivateFormat(index->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, index [%s]",
            ToString(index->GetStorageFormat()).GetString());
    return false;
  }
  if (op::IsPrivateFormat(out->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, out [%s]",
            ToString(out->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *index, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, index, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, index, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline int64_t GetViewShapeSize(const aclTensor *self) {
  int64_t size = 1;
  for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
    size *= self->GetViewShape().GetDim(i);
  }
  return size;
}

aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor *self, const aclTensor *index, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTake, DFX_IN(self, index), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, index, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty() || index->IsEmpty()) {
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

  // 将self转成一维
  const int64_t selfSize[] = {GetViewShapeSize(self)};
  aclIntArray *oneDimArray = uniqueExecutor.get()->AllocIntArray(selfSize, 1);
  auto self1D = l0op::Reshape(selfContiguous, oneDimArray, uniqueExecutor.get());
  CHECK_RET(self1D != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子Gather进行计算
  auto gatherResult = l0op::GatherV2(self1D, 0, indexContiguous, uniqueExecutor.get(), 0, true);
  CHECK_RET(gatherResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(gatherResult, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTake(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTake);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif