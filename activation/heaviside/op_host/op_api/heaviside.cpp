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
 * \file heaviside.cpp
 * \brief
 */

#include "heaviside.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Heaviside);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

[[maybe_unused]] static bool IsAiCoreSupport(const aclTensor* input)
{
    return CheckType(input->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* HeavisideAiCore(
    const aclTensor* input, const aclTensor* values, const aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(HeavisideAiCore, input, values, output);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Heaviside, OP_INPUT(input, values), OP_OUTPUT(output));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HeavisideAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}

const aclTensor* Heaviside(const aclTensor* input, const aclTensor* values, aclOpExecutor* executor)
{
    const aclTensor* output = executor->AllocTensor(input->GetViewShape(), input->GetDataType());
    return HeavisideAiCore(input, values, output, executor);
}
} // namespace l0op