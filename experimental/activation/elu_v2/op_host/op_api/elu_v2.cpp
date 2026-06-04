/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elu_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(EluV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* EluV2AiCore(
    const aclTensor* self, float alpha, float scale, float inputScale, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(EluV2AiCore, self, alpha, scale, inputScale, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(EluV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(alpha, scale, inputScale));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EluV2AiCore failed."),
        return nullptr);
    return out;
}
const aclTensor* EluV2(const aclTensor* self, float alpha, float scale, float inputScale, aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "EluV2 not supported: dtype=%d. "
                "Supported dtypes: FLOAT16, FLOAT, BF16.",
                static_cast<int>(self->GetDataType()));
        return nullptr;
    }
    aclTensor* out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
    OP_CHECK(
        out != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EluV2: AllocTensor for output failed."),
        return nullptr);

    return EluV2AiCore(self, alpha, scale, inputScale, out, executor);    
}
} // namespace l0op
