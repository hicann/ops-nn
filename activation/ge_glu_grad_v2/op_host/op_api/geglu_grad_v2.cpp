/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "geglu_grad_v2.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GeGluGradV2);

const aclTensor* GeGluGradV2(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    bool activateLeft, aclOpExecutor* executor)
{
    L0_DFX(GeGluGradV2, gradOutput, self, gelu, dim, approximate, activateLeft);

    // 根据输入shape申请输出tensor
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), Format::FORMAT_ND);
    if (gradInput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradIntput tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        GeGluGradV2, OP_INPUT(gradOutput, self, gelu), OP_OUTPUT(gradInput), OP_ATTR(dim, approximate, activateLeft));

    return gradInput;
}

} // namespace l0op
