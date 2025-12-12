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
 * \file hard_swish_grad.cpp
 * \brief
 */

#include "hard_swish_grad.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(HardSwishGrad);

const aclTensor* HardSwishGrad(const aclTensor* gradOutput, const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(HardSwishGrad, gradOutput, self);
    // 根据推导出的输出shape申请输出tensor
    // 通过输入shape推导算子输出shape
    auto hardSwishGradOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HardSwishGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(hardSwishGradOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HardSwishGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return hardSwishGradOut;
}
} // namespace l0op