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
 * \file masked_scatter_with_position.cpp
 * \brief
 */
#include "masked_scatter_with_position.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MaskedScatterWithPosition);

static const aclTensor* MaskedScatterWithPositionAiCore(
    const aclTensor* x, const aclTensor* mask, const aclTensor* position, const aclTensor* updates, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(MaskedScatterWithPositionAiCore, x, mask, position, updates, y);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(MaskedScatterWithPosition, OP_INPUT(x, mask, position, updates), OP_OUTPUT(y));
    OP_CHECK(
        retAicore == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedScatterWithPosition ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}

const aclTensor* MaskedScatterWithPosition(const aclTensor* x, const aclTensor* mask, const aclTensor* position, const aclTensor* updates, aclOpExecutor* executor)
{
    L0_DFX(MaskedScatterWithPosition, x, mask, position, updates);
    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    return MaskedScatterWithPositionAiCore(x, mask, position, updates, y, executor);
}

} // namespace l0op