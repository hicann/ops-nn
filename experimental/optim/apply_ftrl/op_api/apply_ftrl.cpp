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

/*!
 * \file apply_ftrl.cpp
 * \brief ApplyFtrl L0 API implementation (namespace l0op).
 */

#include "apply_ftrl.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ApplyFtrl);

// ApplyFtrl AI Core launcher.
//   - 8 inputs map positionally to ports: var, accum, linear, grad, lr, l1, l2, lr_power.
//   - 1 output port (var) is bound to the SAME varRef tensor -> the kernel writes the
//     new var into var's own GM (in place).
//   - accum / linear have NO output port; the kernel writes their new values back into
//     their input GM directly (Ref write-back), so they appear only in OP_INPUT.
//   - use_locking is NOT registered as an OpDef attribute (no effect on the math and not
//     read by tiling/kernel), so OP_ATTR is intentionally omitted to keep the kernel arg
//     layout (8 in + 1 out + workspace + tiling) intact.
std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> ApplyFtrl(
    const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef, const aclTensor* grad,
    const aclTensor* lr, const aclTensor* l1, const aclTensor* l2, const aclTensor* lrPower, aclOpExecutor* executor)
{
    L0_DFX(ApplyFtrl, varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        ApplyFtrl, OP_INPUT(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower), OP_OUTPUT(varRef));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyFtrl ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
    return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(varRef, accumRef, linearRef);
}

} // namespace l0op
