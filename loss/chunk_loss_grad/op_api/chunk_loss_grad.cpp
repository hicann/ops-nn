/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "chunk_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ChunkLossGrad);
const aclTensor* ChunkLossGrad(const aclTensor* hidden, const aclTensor* weight, const aclTensor* labels, float alpha,
                               const aclTensor* softmaxMax, const aclTensor* softmaxSum, const aclTensor* gradOutput,
                               int64_t ignoreIndex, aclOpExecutor* executor)
{
    auto gradWeightOut = executor->AllocTensor(weight->GetViewShape(), weight->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(gradWeightOut != nullptr, nullptr);
    L0_DFX(ChunkLossGrad, hidden, weight, labels, alpha, softmaxMax, softmaxSum, gradOutput, ignoreIndex);
    auto status = ADD_TO_LAUNCHER_LIST_AICORE(ChunkLossGrad,
                                              OP_INPUT(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput),
                                              OP_OUTPUT(gradWeightOut), OP_ATTR(alpha, ignoreIndex));
    OP_CHECK(status == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ChunkLossGrad launch planning failed."),
             return nullptr);
    return gradWeightOut;
}
} // namespace l0op
