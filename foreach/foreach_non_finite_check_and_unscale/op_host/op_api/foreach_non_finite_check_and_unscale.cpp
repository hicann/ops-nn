/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "foreach_non_finite_check_and_unscale.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ForeachNonFiniteCheckAndUnscale);

aclnnStatus ForeachNonFiniteCheckAndUnscale(const aclTensorList* scaledGrads, const aclTensor* foundInf,
                                            const aclTensor* invScale, aclOpExecutor* executor)
{
    L0_DFX(ForeachNonFiniteCheckAndUnscale, scaledGrads, foundInf, invScale);
    // All three parameters are inputs in the OpDef; the first two are updated in place.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ForeachNonFiniteCheckAndUnscale, OP_INPUT(scaledGrads, foundInf, invScale),
                                           OP_OUTPUT());
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ret, "Failed to add ForeachNonFiniteCheckAndUnscale to the launcher list.");
    }
    return ret;
}
} // namespace l0op
