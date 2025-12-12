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
 * \file hardtanh_grad.cpp
 * \brief
 */

#include "hardtanh_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(HardtanhGrad);

const aclTensor *HardtanhGrad(const aclTensor *gradOutput, const aclTensor *self,
                              float min, float max,
                              const aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(HardtanhGrad, gradOutput, self, min, max, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HardtanhGrad,
                                         OP_INPUT(self, gradOutput),
                                         OP_OUTPUT(out),
                                         OP_ATTR(min, max));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardtanhGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op