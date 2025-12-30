/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_update_scatter.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(QuantUpdateScatter);

const aclTensor* QuantUpdateScatter(const aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates,
                                    const aclTensor* quantScales, const aclTensor* quantZeroPoints,
                                    const std::string& reduce, int64_t axis, int64_t quantAxis,
                                    aclOpExecutor* executor, bool reciprocalScale, const char* roundMode) {
  L0_DFX(QuantUpdateScatter, selfRef, indices, updates, quantScales, quantZeroPoints,
         reduce, axis, quantAxis, reciprocalScale, roundMode);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantUpdateScatter, OP_INPUT(selfRef, indices, updates, quantScales, quantZeroPoints),
                                         OP_OUTPUT(selfRef), OP_ATTR(reduce, axis, quantAxis, reciprocalScale, roundMode));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantUpdateScatter ADD_TO_LAUNCHER_LIST_AICORE failed."),
                                         return nullptr);                                         
  return selfRef;
}
}  // namespace l0op
