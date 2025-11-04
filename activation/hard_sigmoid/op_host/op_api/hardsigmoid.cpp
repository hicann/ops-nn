/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hardsigmoid.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {
    constexpr float kHardSigmoidAlpha = 1.0f / 6.0f;
    constexpr float kHardSigmoidBeta = 0.5f;
}

namespace l0op {
OP_TYPE_REGISTER(HardSigmoid);
const aclTensor* HardSigmoid(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(HardSigmoid, self);
    auto hardsigmoidOut = executor->AllocTensor(self->GetStorageShape(), self->GetDataType(), self->GetStorageFormat());
    CHECK_RET(hardsigmoidOut != nullptr, nullptr);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(HardSigmoid, OP_INPUT(self), OP_ATTR(kHardSigmoidAlpha, kHardSigmoidBeta), OP_OUTPUT(hardsigmoidOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardSigmoidAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return hardsigmoidOut;
}
} // namespace l0op