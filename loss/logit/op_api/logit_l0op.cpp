/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logit_l0op.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Logit);

const aclTensor* Logit(const aclTensor* input, float eps, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(Logit, input, eps, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Logit, OP_INPUT(input), OP_OUTPUT(out), OP_ATTR(eps));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Logit ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}
} // namespace l0op
