/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_inverse.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MatrixInverse);

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

// 根据芯片类型、dtype判断算子是否支持数据类型
static inline bool IsSupport(const aclTensor *self) {
  // MatrixInverse只需要判断dtype
  return CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST);
}

static inline const aclTensor *MatrixInverseAicpu(const aclTensor *self, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(MatrixInverseAicpu, self, out)

  static internal::AicpuTaskSpace space("MatrixInverse", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(MatrixInverse, OP_ATTR_NAMES({"adjoint"}), OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(false));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

  return out;
}

const aclTensor *MatrixInverse(const aclTensor *self, aclOpExecutor *executor) {
  auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc out tensor failed.");
    return nullptr;
  }

  if (!IsSupport(self)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s should be in dtype support list %s.",
            ToString(self->GetDataType()).GetString(), ToString(DTYPE_SUPPORT_LIST).GetString());
    return nullptr;
  }

  return MatrixInverseAicpu(self, out, executor);
}
} // l0op
