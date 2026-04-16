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
 * @file fast_gelu_v2.cpp
 * @brief ACLNN L0 API implementation for FastGeluV2
 */

#include "fast_gelu_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FastGeluV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* x)
{
    return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool FastGeluV2InferShape(const op::Shape& xShape, op::Shape& outShape)
{
    outShape = xShape;
    return true;
}

static const aclTensor* FastGeluV2AiCore(const aclTensor* x, const aclTensor* out,
                                          aclOpExecutor* executor)
{
    L0_DFX(FastGeluV2AiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FastGeluV2,
        OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FastGeluV2AiCore failed."),
        return nullptr);
    return out;
}

const aclTensor* FastGeluV2(const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!FastGeluV2InferShape(x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "FastGeluV2 not supported: dtype x=%d. Supported: FLOAT, FLOAT16, BF16.",
                static_cast<int>(x->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, x->GetDataType());

    return FastGeluV2AiCore(x, out, executor);
}

} // namespace l0op
