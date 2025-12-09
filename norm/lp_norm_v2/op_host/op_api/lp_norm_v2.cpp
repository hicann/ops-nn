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
 * \file lp_norm_v2.cpp
 * \brief
 */
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "lp_norm_v2.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LpNormV2);

const aclTensor* LpNormV2(
    const aclTensor* x, float p, const aclIntArray* dims, bool keepDim, float epsilon, aclOpExecutor* executor)
{
    L0_DFX(LpNormV2, x, p, dims, keepDim, epsilon);
    auto out = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    INFER_SHAPE(LpNormV2, OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(p, dims, keepDim, epsilon));

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LpNormV2, OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(p, dims, keepDim, epsilon));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LpNormV2 ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
} // namespace l0op