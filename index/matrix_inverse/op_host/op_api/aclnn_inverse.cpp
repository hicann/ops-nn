/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_inverse.h"
#include "matrix_inverse.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr int64_t MAX_DIM_LEN = 8;
static constexpr int64_t MIN_DIM_LEN = 2;
static constexpr int64_t LAST_DIM = 1;
static constexpr int64_t LAST_2_DIM = 2;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_FLOAT16};

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  // self、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // self的数据类型能否转换为out的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  if (!self->IsEmpty()) {
    // self的数据维度不能小于2
    OP_CHECK_MIN_DIM(self, MIN_DIM_LEN, return false);

    const Shape &shape = self->GetViewShape();
    const int64_t dimNum = static_cast<int64_t>(shape.GetDimNum());
    // self必须是方阵
    if (shape.GetDim(dimNum - LAST_DIM) != shape.GetDim(dimNum - LAST_2_DIM)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "The input tensor must be batches of square matrices, but they are %ld by %ld matrices",
              shape.GetDim(dimNum - LAST_DIM), shape.GetDim(dimNum - LAST_2_DIM));
      return false;
    }
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInverse, DFX_IN(self), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (self->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // self如果非连续，需要转连续
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果self是FP16，则需要转换成FP32
  auto castSelf = contiguousSelf;
  if (self->GetDataType() == DataType::DT_FLOAT16) {
    castSelf = l0op::Cast(contiguousSelf, DataType::DT_FLOAT, uniqueExecutor.get());
  }
  CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子MatrixInverse进行计算
  auto matrixInverseOut = l0op::MatrixInverse(castSelf, uniqueExecutor.get());
  CHECK_RET(matrixInverseOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(matrixInverseOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInverse);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}