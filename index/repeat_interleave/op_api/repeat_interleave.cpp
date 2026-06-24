/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave.cpp
 * \brief
 */
#include "repeat_interleave.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RepeatInterleave);
OP_TYPE_REGISTER(RepeatInterleaveV2);

static op::Shape RepeatInterleaveInferShape(const aclTensor* x, int64_t axis, int64_t outputSize)
{
    // 运行repeatInterleave的轴一定是dim。对应的dim维度大小必须为outputSize
    op::Shape initialShape = x->GetViewShape();
    initialShape.SetDim(axis, outputSize);
    return initialShape;
}

// 只支持AICORE
const aclTensor* RepeatInterleave(
    const aclTensor* x, const aclTensor* repeats, int64_t axis, int64_t outputSize, aclOpExecutor* executor)
{
    L0_DFX(RepeatInterleave, x, repeats, axis, outputSize);
    op::Shape outShape = RepeatInterleaveInferShape(x, axis, outputSize);
    auto out = executor->AllocTensor(outShape, x->GetDataType(), x->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleave, OP_INPUT(x, repeats), OP_OUTPUT(out), OP_ATTR(axis));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatInterleaveAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

// 只支持AICORE
const aclTensor* RepeatInterleaveV2(
    const aclTensor* x, const aclTensor* repeats, int64_t axis, int64_t outputSize, aclOpExecutor* executor)
{
    L0_DFX(RepeatInterleaveV2, x, repeats, axis, outputSize);
    op::Shape outShape = RepeatInterleaveInferShape(x, axis, outputSize);
    auto out = executor->AllocTensor(outShape, x->GetDataType(), x->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RepeatInterleaveV2, OP_INPUT(x, repeats), OP_OUTPUT(out), OP_ATTR(axis));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatInterleaveV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
} // namespace l0op