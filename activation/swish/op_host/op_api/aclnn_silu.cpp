/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_silu.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "swish.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base_caculation.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  CHECK_RET(CheckDtypeValidActivation(self, out, supportList), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckSameShapeNotlimit1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSiluGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSilu, DFX_IN(self), DFX_OUT(out));

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

  auto shapeOri = self->GetViewShape();
  int64_t dimSize = self->GetViewShape().GetDimNum();
  auto shapeOriDetial = GetTensorShapeActivation(selfContiguous, uniqueExecutor.get());
  auto reshapeSelf = selfContiguous;

  if (dimSize > (int64_t)MAX_SUPPORT_DIMS_NUMS) {
    int64_t AllDimValue = 1;
    for (int i = 0; i < dimSize; i++) {
      AllDimValue *= shapeOri[i];
    }
    int64_t AllDim[1] = {AllDimValue};
    auto shape1d = (uniqueExecutor)->AllocIntArray(AllDim, 1);
    reshapeSelf = ReshapeLongTensorActivation(selfContiguous, uniqueExecutor.get(), dimSize, shape1d);
  }

  float scale = 1.0;
  auto siluOut = l0op::Swish(reshapeSelf, scale, uniqueExecutor.get());
  CHECK_RET(siluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reshapeSiluOut = siluOut;
  if (dimSize > (int64_t)MAX_SUPPORT_DIMS_NUMS) {
    reshapeSiluOut = ReshapeLongTensorActivation(siluOut, uniqueExecutor.get(), dimSize, shapeOriDetial);
  }

  auto viewCopyOut = l0op::ViewCopy(reshapeSiluOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSilu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSilu);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
