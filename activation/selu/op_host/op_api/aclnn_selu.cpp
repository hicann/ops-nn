/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_selu.h"

#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "selu.h"
#include "aclnn_kernels/transdata.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static inline const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8};

static inline const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT8, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline aclIntArray *GetTensorShape(const aclTensor *x, aclOpExecutor *executor) {
  auto shape = x->GetViewShape();
  int64_t dimSize = x->GetViewShape().GetDimNum();

  std::vector<int64_t> valuePerm(dimSize);
  for (int i = 0; i < dimSize; i++) {
    valuePerm[i] = shape[i];
  }
  auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
  return perm;
}

static const aclTensor *ReshapeLongTensor(const aclTensor *x, aclOpExecutor *executor, int originalDimSize,
                                          aclIntArray *valuePerm = nullptr) {
  int64_t dimSize = x->GetViewShape().GetDimNum();
  if (originalDimSize == dimSize && dimSize <= (int64_t)MAX_SUPPORT_DIMS_NUMS) {
    return x;
  }

  auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
  return reshapeSelf;
}

aclnnStatus ExecSeluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  size_t dimSize = self->GetViewShape().GetDimNum();
  auto shapeOriDetail = GetTensorShape(selfContiguous, uniqueExecutor.get());
  auto reshapeSelf = selfContiguous;

  if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
    auto allDimValue = self->Size();
    int64_t allDim[1] = {allDimValue};
    auto shape1d = (uniqueExecutor)->AllocIntArray(allDim, 1);
    reshapeSelf = ReshapeLongTensor(selfContiguous, uniqueExecutor.get(), dimSize, shape1d);
  }
  auto seluOut = l0op::Selu(reshapeSelf, uniqueExecutor.get());
  CHECK_RET(seluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reshapeSeluOut = seluOut;
  if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
    reshapeSeluOut = ReshapeLongTensor(seluOut, uniqueExecutor.get(), dimSize, shapeOriDetail);
  }
  auto castSeluOut = l0op::Cast(reshapeSeluOut, out->GetDataType(), uniqueExecutor.get());
  auto viewCopyOut = l0op::ViewCopy(castSeluOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSeluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                      aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnSelu, DFX_IN(self), DFX_OUT(out));
  return ExecSeluGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSeluGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceSelu, DFX_IN(selfRef), DFX_OUT(selfRef));
  auto out = const_cast<aclTensor *>(selfRef);
  return ExecSeluGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnSelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSelu);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                             aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceSelu);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
