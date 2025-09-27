/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "relu_grad.h"
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
OP_TYPE_REGISTER(ReluGrad);

const aclTensor *ReluGrad(const aclTensor *gradOutput, const aclTensor *self, aclOpExecutor *executor) {
  L0_DFX(ReluGrad, gradOutput, self);
  // 根据推导出的输出shape申请输出tensor
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape, return nullptr);
  // 第一个参数是输出shape，第二个参数是输出的dtype
  auto out = executor->AllocTensor(broadcastShape, self->GetDataType());

  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ReluGrad,
                                               OP_INPUT(gradOutput, self),
                                               OP_OUTPUT(out));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "ReluGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}
}  // namespace l0op