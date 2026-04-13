/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
  * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file softsign.cpp
 * @brief ACLNN L0 API implementation for Softsign operator
 *
 * L0 API: shape inference, kernel dispatch
 * L2 API: parameter checking, Contiguous/ViewCopy handling
 */

#include "softsign.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Softsign);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* SoftsignAiCore(const aclTensor* self,
                                        const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(SoftsignAiCore, self, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Softsign,
        OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftsignAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API entry point
 *
 * Flow:
 * 1. IsAiCoreSupport - check execution path
 * 2. AllocTensor     - allocate output tensor
 * 3. SoftsignAiCore  - call kernel
 */
const aclTensor* Softsign(const aclTensor* self, aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Softsign not supported: dtype=%d. "
                "Supported dtypes: FLOAT16, FLOAT, BF16.",
                static_cast<int>(self->GetDataType()));
        return nullptr;
    }

    // Output shape = input shape (elementwise), output dtype = input dtype
    const aclTensor* out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    OP_CHECK(
        out != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Softsign: AllocTensor for output failed."),
        return nullptr);

    return SoftsignAiCore(self, out, executor);
}

} // namespace l0op
