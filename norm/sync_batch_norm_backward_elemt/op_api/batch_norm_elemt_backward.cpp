/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "batch_norm_elemt_backward.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SyncBatchNormBackwardElemt);

const aclTensor* SyncBatchNormBackwardElemt(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* meadDy, const aclTensor* meadDyXmu, aclOpExecutor* executor)
{
    L0_DFX(SyncBatchNormBackwardElemt, gradOut, input, mean, invstd, weight, meadDy, meadDyXmu);

    auto gradInput = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), input->GetViewFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBatchNormBackwardElemt, OP_INPUT(gradOut, input, mean, invstd, weight, meadDy, meadDyXmu),
        OP_OUTPUT(gradInput));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SyncBatchNormBackwardElemtAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gradInput;
}
} // namespace l0op