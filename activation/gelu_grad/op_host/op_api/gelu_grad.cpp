/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_grad.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(GeluGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* unused, const aclTensor* gradInput)
{
    auto checkGradOutputType = CheckType(gradOutput->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    auto checkOutputType = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    auto checkGradInputType = CheckType(gradInput->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    auto checkUnusedType = CheckType(unused->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    return (checkGradOutputType && checkOutputType && checkGradInputType && checkUnusedType);
}

// AiCore逻辑
static void GeluGradAiCore(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* unused, const aclTensor* gradInput,
    aclOpExecutor* executor)
{
    L0_DFX(GeluGradAiCore, gradOutput, self, unused, gradInput);

    // GeluGrad的input的第三个参数unused，默认传gradOutput
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(GeluGrad, OP_INPUT(gradOutput, self, unused), OP_OUTPUT(gradInput));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GeluGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
}

const aclTensor* GeluGrad(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* unused, const aclTensor* gradInput,
    aclOpExecutor* executor)
{
    L0_DFX(GeluGrad, gradOutput, self, unused, gradInput);
    // 判断走AiCore还是AiCPU，目前无AiCPU实现，默认走AiCore
    if (!IsAiCoreSupport(gradOutput, self, unused, gradInput)) {
        return nullptr;
    }
    GeluGradAiCore(gradOutput, self, unused, gradInput, executor);
    return gradInput;
}
} // namespace l0op
