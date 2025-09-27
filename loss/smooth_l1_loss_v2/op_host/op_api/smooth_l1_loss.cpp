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
 * \file smooth_l1_loss.cpp
 * \brief
 */

#include "smooth_l1_loss.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SmoothL1LossV2);

const aclTensor *SmoothL1Loss(const aclTensor *self, const aclTensor *target, const std::string &reduction, float beta,
                              aclOpExecutor *executor) {
  L0_DFX(SmoothL1Loss, self, target, reduction, beta, reduction);
  auto result = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());
  auto ret = INFER_SHAPE(SmoothL1LossV2, OP_INPUT(self, target), OP_OUTPUT(result),
                         OP_ATTR(beta, reduction.c_str()));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SmoothL1LossV2 InferShape failed.");
    return nullptr;
  }
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SmoothL1LossV2,
                                               OP_INPUT(self, target),
                                               OP_OUTPUT(result),
                                               OP_ATTR(beta, reduction.c_str()));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "SmoothL1LossV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");

  return result;
}
}  // namespace l0op
