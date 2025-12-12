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
 * \file foreach_addcdiv_scalar_v2.cpp
 * \brief
 */

#include "foreach_addcdiv_scalar_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ForeachAddcdivScalar);

const aclTensorList *ForeachAddcdivScalarV2(const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3,
                                            const aclTensor *scalar, const aclTensorList *out, aclOpExecutor *executor) {
    L0_DFX(ForeachAddcdivScalarV2, x1, x2, x3, scalar);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ForeachAddcdivScalar,
                                      OP_INPUT(x1, x2, x3, scalar),
                                      OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}
}  // namespace l0op

