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
 * \file inplace_scatter_add.cpp
 * \brief
 */
#include "inplace_scatter_add.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(InplaceScatterAdd);
// AICORE算子kernel
const aclTensor *InplaceScatterAddAiCore(const aclTensor *self, const aclTensor *index,
                                       const aclTensor *source, aclOpExecutor *executor) {
  L0_DFX(InplaceScatterAddAiCore, self, index, source);
  auto scatterAddOut = const_cast<aclTensor*>(self);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceScatterAdd,
                                         OP_INPUT(self, index, source),
                                         OP_OUTPUT(scatterAddOut));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "InplaceScatterAddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return scatterAddOut;
}
}  // namespace l0op
