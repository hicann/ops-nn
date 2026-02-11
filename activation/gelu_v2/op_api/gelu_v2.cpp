/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_v2.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(GeluV2);

const aclTensor* GeluV2(const aclTensor* x, const std::string& approximate, aclOpExecutor* executor)
{
    L0_DFX(GeluV2, x, approximate);
    // 通过输入推导出输出
    auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(GeluV2, OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(approximate));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GeluV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "GeluV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op