/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "threshold_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ThresholdGradV2D);

const aclTensor *ThresholdGradV2D(const aclTensor *gradOutput, const aclTensor *self, float threshold,
                                  aclOpExecutor *executor) {
  L0_DFX(ThresholdGradV2D, gradOutput, self, threshold);
  // 根据推导出的输出shape申请输出tensor
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return nullptr);
  // 第一个参数是输出shape，第二个参数是输出的dtype
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());

  ADD_TO_LAUNCHER_LIST_AICORE(ThresholdGradV2D,
                              OP_INPUT(gradOutput, self),
                              OP_OUTPUT(out),
                              OP_ATTR(threshold));
  return out;
}
}  // namespace l0op