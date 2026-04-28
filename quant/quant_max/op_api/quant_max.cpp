/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_max.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantMax);

static const aclTensor* QuantMaxAiCore(
    const aclTensor* x, const aclTensor* scale, const char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* amax, aclOpExecutor* executor)
{
    L0_DFX(QuantMaxAiCore, x, scale, roundMode, dstType, y, amax);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(QuantMax, OP_INPUT(x, scale), OP_OUTPUT(y, amax), OP_ATTR(roundMode, dstType));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantMaxAiCore failed."), return nullptr);
    return y;
}

const aclTensor* QuantMax(
    const aclTensor* x, const aclTensor* scale, const char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* amax, aclOpExecutor* executor)
{
    return QuantMaxAiCore(x, scale, roundMode, dstType, y, amax, executor);
}

} // namespace l0op
