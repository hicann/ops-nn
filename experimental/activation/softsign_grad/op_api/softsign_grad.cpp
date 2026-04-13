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
 * @file softsign_grad.cpp
 * @brief ACLNN L0 API 实现 - SoftsignGrad
 *
 * L0 API 职责：形状推导、Kernel 调度
 */

#include "softsign_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SoftsignGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* gradients, const aclTensor* features)
{
    return CheckType(gradients->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           CheckType(features->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// SoftsignGrad 是 Elementwise 算子，output shape = input shape，不需要广播
static bool SoftsignGradInferShape(const op::Shape& gradShape, op::Shape& outShape)
{
    outShape = gradShape;
    return true;
}

static const aclTensor* SoftsignGradAiCore(
    const aclTensor* gradients,
    const aclTensor* features,
    const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(SoftsignGradAiCore, gradients, features, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SoftsignGrad,
        OP_INPUT(gradients, features), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftsignGradAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape - 输出 shape = 输入 shape
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor - 分配输出 Tensor
 * 4. SoftsignGradAiCore - 调用 Kernel
 */
const aclTensor* SoftsignGrad(
    const aclTensor* gradients,
    const aclTensor* features,
    aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    if (!SoftsignGradInferShape(gradients->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SoftsignGrad infer shape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(gradients, features)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SoftsignGrad not supported: dtype gradients=%d, features=%d. "
                "Supported: FLOAT16, FLOAT, BF16.",
                static_cast<int>(gradients->GetDataType()),
                static_cast<int>(features->GetDataType()));
        return nullptr;
    }

    out = executor->AllocTensor(outShape, gradients->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SoftsignGrad AllocTensor for output failed");
        return nullptr;
    }

    return SoftsignGradAiCore(gradients, features, out, executor);
}

} // namespace l0op
