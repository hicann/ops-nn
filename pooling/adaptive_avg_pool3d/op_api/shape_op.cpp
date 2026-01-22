/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shape_op.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

namespace l0op {
OP_TYPE_REGISTER(Shape_op);
const aclTensor *Shape_op(const aclTensor *x, aclOpExecutor *executor) {
  L0_DFX(Shape_op, x);
  auto input_shape = x->GetViewShape();
  size_t dim_num = input_shape.GetDimNum();
  int64_t sizes[dim_num];
  for (size_t i = 0; i < dim_num; ++i) {
    sizes[i] = input_shape.GetDim(i);
  }
  aclIntArray *shapes = executor->AllocIntArray(sizes, dim_num);
  auto res = executor->ConvertToTensor(shapes, op::DataType::DT_INT64);
  return res;
}
}  // namespace l0op