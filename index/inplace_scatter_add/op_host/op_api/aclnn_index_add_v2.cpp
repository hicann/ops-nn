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
 * \file aclnn_index_add_v2.cpp
 * \brief
 */

#include "aclnn_index_add_v2.h"
#include "aclnn_index_add.h"
#include "inplace_scatter_add.h"
#include "level0/inplace_index_add.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
static constexpr size_t MAX_DIM_LEN = 8;
static constexpr size_t MODE_0_SUPPORT_DIM = 2;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_BF16};

static bool IsAICoreSupport(const aclTensor *self, const int64_t dim, const aclTensor *index) {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    if (!CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
      return false;
    }
  }
  int64_t selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  int64_t indexDimNum = static_cast<int64_t>(index->GetViewShape().GetDimNum());
  if(selfDimNum != MODE_0_SUPPORT_DIM || indexDimNum != 1) {
    return false;
  }
  if(!(dim == 0 || dim == 0 - MODE_0_SUPPORT_DIM)) {
    return false;
  }

  return true;
}

static bool CheckNotNull(const aclTensor * self, const aclTensor * index, const aclTensor * source,
                         const aclScalar * alpha, const aclTensor * out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(source, return false);
  OP_CHECK_NULL(alpha, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const op::DataType selfDtype, const op::DataType indexDtype,
                            const op::DataType sourceDtype, const op::DataType outDtype, const op::DataType alphaDtype) {
  if (!CheckType(selfDtype, ASCEND910B_DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s should be in dtype support list [%s].",
            op::ToString(selfDtype).GetString(), op::ToString(ASCEND910B_DTYPE_SUPPORT_LIST).GetString());
    return false;
  }
  if (!CheckType(indexDtype, INDEX_DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index dtype %s should be in dtype support list [%s].",
            op::ToString(indexDtype).GetString(), op::ToString(INDEX_DTYPE_SUPPORT_LIST).GetString());
    return false;
  }

  if (selfDtype != sourceDtype) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and source dtype %s must same.",
            op::ToString(selfDtype).GetString(), op::ToString(sourceDtype).GetString());
    return false;
  }

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(alphaDtype, selfDtype, return false);

  if (selfDtype != outDtype) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and source dtype %s must same.",
            op::ToString(selfDtype).GetString(), op::ToString(outDtype).GetString());
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *index, const aclTensor *source,
                       const aclTensor *out, int64_t dim) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(source, MAX_DIM_LEN, return false);
  int64_t selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  if ((dim < -selfDimNum || dim >= selfDimNum) && (
      !(dim == 0 && selfDimNum == 0)
      ))  {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in [-self.dims(), self.dims), but dim is %ld.", dim);
    return false;
  }
  if (self->GetViewShape().GetDimNum() != 0 && self->GetViewShape().GetDimNum() != source->GetViewShape().GetDimNum()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DimNum of self must be equal to dimNum of source.");
    return false;
  }
  int64_t dimStd = dim >= 0 ? dim : self->GetViewShape().GetDimNum() + dim;
  for (size_t idx = 0; idx < self->GetViewShape().GetDimNum(); idx++) {
    if (dimStd != static_cast<int64_t>(idx)) {
      if (self->GetViewShape().GetDim(idx) != source->GetViewShape().GetDim(idx)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of self must match the size of source at dimension %ld.", dim);
        return false;
      }
    }
  }
  if (index->GetViewShape().GetDimNum() > 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index dimension %zu is supposed to be 1.", index->GetViewShape().GetDimNum());
    return false;
  }
  if (index->GetViewShape().GetDim(0) != source->GetViewShape().GetDim(dimStd)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Number of index should be equal to number of source in dim  %ld.", dim);
    return false;
  }

  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const int64_t dim, const aclTensor *index,
                               const aclTensor *source, const aclScalar *alpha, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, source, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self->GetDataType(), index->GetDataType(), source->GetDataType(),
                            out->GetDataType(), alpha->GetDataType()), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self, index, source, out, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor *UnDeterministicImpl(const aclTensor *selfContiguous, const int64_t dim,
                                     const aclTensor *indexContiguous, const aclTensor *sourceContiguous,
                                     const aclScalar *alpha, int64_t mode, aclOpExecutor *executor) {
  const aclTensor* indexAddOut = nullptr;
  if(mode == 0) {
    if (!IsAICoreSupport(selfContiguous, dim, indexContiguous)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support mode == 0, please check!");
      return nullptr;
    }
    const aclTensor* selfContiguousCast = nullptr;
    const aclTensor* sourceContiguousCast = nullptr;
    if (selfContiguous->GetDataType() == op::DataType::DT_INT16 ||
        selfContiguous->GetDataType() == op::DataType::DT_INT32) {
      selfContiguousCast = l0op::Cast(selfContiguous, op::DataType::DT_INT32, executor);
      sourceContiguousCast = l0op::Cast(sourceContiguous, op::DataType::DT_INT32, executor);
    } else {
      selfContiguousCast = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
      sourceContiguousCast = l0op::Cast(sourceContiguous, op::DataType::DT_FLOAT, executor);
    }
    indexAddOut = l0op::InplaceScatterAddAiCore(selfContiguousCast, indexContiguous,
                                                sourceContiguousCast, executor);
    CHECK_RET(indexAddOut != nullptr, nullptr);
  } else {
    const aclTensor* selfContiguousCast = nullptr;
    const aclTensor* sourceContiguousCast = nullptr;
    if (selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 ||
        selfContiguous->GetDataType() == op::DataType::DT_FLOAT ||
        selfContiguous->GetDataType() == op::DataType::DT_BF16) {
        selfContiguousCast = l0op::Cast(selfContiguous, op::DataType::DT_DOUBLE, executor);
        sourceContiguousCast = l0op::Cast(sourceContiguous, op::DataType::DT_DOUBLE, executor);
    } else {
        selfContiguousCast = l0op::Cast(selfContiguous, selfContiguous->GetDataType(), executor);
        sourceContiguousCast = l0op::Cast(sourceContiguous, sourceContiguous->GetDataType(), executor);
    }
    const aclTensor* alphaTensor = executor->ConvertToTensor(alpha, selfContiguousCast->GetDataType());
    CHECK_RET(alphaTensor != nullptr, nullptr);
    indexAddOut = l0op::InplaceIndexAddAiCpu(selfContiguousCast, dim, indexContiguous, sourceContiguousCast, alphaTensor,
                                             executor);
    CHECK_RET(indexAddOut != nullptr, nullptr);
  }
  return indexAddOut;
}

static aclnnStatus ExecAclnnIndexAddV2GetWorkspaceSize(const aclTensor *self, const int64_t dim, const aclTensor *index,
    const aclTensor *source, const aclScalar *alpha, int64_t mode, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  if (index != nullptr && index->GetViewShape().GetDimNum() == 0) {
    int64_t zero = 0;
    index = l0op::UnsqueezeNd(index, zero, uniqueExecutor.get());
    CHECK_RET(index != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (source != nullptr && source->GetViewShape().GetDimNum() == 0) {
    int64_t zero = 0;
    source = l0op::UnsqueezeNd(source, zero, uniqueExecutor.get());
    CHECK_RET(source != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto ret = CheckParams(self, dim, index, source, alpha, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (self->IsEmpty() || source->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
  CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* indexAddOut = nullptr;
  indexAddOut = UnDeterministicImpl(selfContiguous, dim, indexContiguous, sourceContiguous,
                                    alpha, mode, uniqueExecutor.get());
  CHECK_RET(indexAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* indexAddOutCast = l0op::Cast(indexAddOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(indexAddOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto viewCopyResult = l0op::ViewCopy(indexAddOutCast, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor *self, const int64_t dim, const aclTensor *index,
    const aclTensor *source, const aclScalar *alpha, int64_t mode, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  // 固定写法，参数检查
  L2_DFX_PHASE_1(aclnnIndexAddV2, DFX_IN(self, dim, index, source, alpha, mode), DFX_OUT(out));
  int64_t deterministicValue = 0;
  rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
  if (retRts != RT_ERROR_NONE) {
      deterministicValue = 0;
  }

  if (deterministicValue == 0) {
    return ExecAclnnIndexAddV2GetWorkspaceSize(self, dim, index, source, alpha, mode, out, workspaceSize, executor);
  } else {
    return aclnnIndexAddGetWorkspaceSize(self, dim, index, source, alpha, out, workspaceSize, executor);
  }
}

aclnnStatus aclnnIndexAddV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                          aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnIndexAddV2);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
