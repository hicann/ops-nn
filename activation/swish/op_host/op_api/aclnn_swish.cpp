/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_swish.h"

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

static bool CheckDtypeValidBetaToFloat(const aclScalar* betaOptional) {
  // 检查betaOptional的数据类型能否转换为FLOAT
  if (betaOptional != nullptr && !CanCast(betaOptional->GetDataType(), DataType::DT_FLOAT)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "betaOptional dtype %s can not cast to float32.",
    ToString(betaOptional->GetDataType()).GetString());
    return false;
  }
  return true;
}

static bool CheckDim(const aclTensor *self, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar* betaOptional, const aclTensor *out) {
  CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

  auto supportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
  CHECK_RET(CheckDtypeValidActivation(self, out, supportList), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckDtypeValidBetaToFloat(betaOptional), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckDim(self, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckSameShapeNotlimit1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor *reshapeLongTensor(const aclTensor *x, aclOpExecutor *executor, size_t originalDimSize,
                                          aclIntArray *valuePerm = nullptr) {
  size_t dimSize = x->GetViewShape().GetDimNum();
  if (originalDimSize == dimSize && dimSize <= MAX_SUPPORT_DIMS_NUMS) {
    return x;
  }

  auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
  return reshapeSelf;
}

aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor* self, const aclScalar* betaOptional, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnSwish, DFX_IN(self, betaOptional), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, betaOptional, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || out->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  size_t dimSize = self->GetViewShape().GetDimNum();
  auto shapeOriDetial = GetTensorShapeActivation(selfContiguous, uniqueExecutor.get());
  auto reshapeSelf = ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor);

  float scale = 1.0f;
  if (betaOptional != nullptr) {
    scale = betaOptional->ToFloat();
  }

  auto swishOut = l0op::Swish(reshapeSelf, scale, uniqueExecutor.get());
  CHECK_RET(swishOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto reshapeSwishOut = swishOut;
  if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
    reshapeSwishOut = reshapeLongTensor(swishOut, uniqueExecutor.get(), dimSize, shapeOriDetial);
  }

  auto viewCopyOut = l0op::ViewCopy(reshapeSwishOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSwish(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSwish);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
