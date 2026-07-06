/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"
#include "group_norm_l0.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupNorm);

static std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNormAICore(const aclTensor* x, const aclTensor* gamma,
                                                                      const aclTensor* beta, int64_t numGroups,
                                                                      float eps, aclTensor* y, aclTensor* mean,
                                                                      aclTensor* rstd, aclOpExecutor* executor)
{
    L0_DFX(GroupNormAICore, x, gamma, beta, numGroups, eps, y, mean, rstd);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupNorm, OP_INPUT(x, gamma, beta), OP_OUTPUT(y, mean, rstd),
                                           OP_ATTR(numGroups, eps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr),
                                         "GroupNorm add to aicore launch list failed.");
    return std::tie(y, mean, rstd);
}

const std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNorm(const aclTensor* x, const aclTensor* gamma,
                                                               const aclTensor* beta, int64_t n, int64_t numGroups,
                                                               float eps, aclOpExecutor* executor)
{
    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    auto mean = executor->AllocTensor(op::Shape({n, numGroups}), x->GetDataType(), op::Format::FORMAT_ND);
    auto rstd = executor->AllocTensor(op::Shape({n, numGroups}), x->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(y != nullptr && mean != nullptr && rstd != nullptr, std::tuple(nullptr, nullptr, nullptr));
    return GroupNormAICore(x, gamma, beta, numGroups, eps, y, mean, rstd, executor);
}
} // namespace l0op
