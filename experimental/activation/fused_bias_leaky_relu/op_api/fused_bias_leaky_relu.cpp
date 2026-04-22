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
 * @file fused_bias_leaky_relu.cpp
 * @brief ACLNN L0 API implementation for FusedBiasLeakyRelu
 *
 * L0 API responsibilities: shape inference, kernel dispatch
 * L2 API responsibilities: parameter checks, Contiguous/ViewCopy
 */

#include "fused_bias_leaky_relu.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedBiasLeakyRelu);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* x, const aclTensor* bias)
{
    return CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           CheckType(bias->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool FusedBiasLeakyReluInferShape(const op::Shape& xShape, op::Shape& outShape)
{
    // Elementwise: output shape = input x shape
    outShape = xShape;
    return true;
}

static const aclTensor* FusedBiasLeakyReluAiCore(const aclTensor* x, const aclTensor* bias,
                                                    double negativeSlope, double scale,
                                                    const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FusedBiasLeakyReluAiCore, x, bias, out);

    // Cast double to float to match OpDef attr type (Float)
    float fNegativeSlope = static_cast<float>(negativeSlope);
    float fScale = static_cast<float>(scale);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedBiasLeakyRelu,
        OP_INPUT(x, bias), OP_OUTPUT(out),
        OP_ATTR(fNegativeSlope, fScale));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedBiasLeakyReluAiCore failed."),
        return nullptr);
    return out;
}

const aclTensor* FusedBiasLeakyRelu(const aclTensor* x, const aclTensor* bias,
                                     double negativeSlope, double scale,
                                     aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!FusedBiasLeakyReluInferShape(x->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(x, bias)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "FusedBiasLeakyRelu not supported: dtype x=%d, bias=%d. "
                "Supported: FLOAT16, FLOAT.",
                static_cast<int>(x->GetDataType()), static_cast<int>(bias->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, x->GetDataType());

    return FusedBiasLeakyReluAiCore(x, bias, negativeSlope, scale, out, executor);
}

} // namespace l0op
