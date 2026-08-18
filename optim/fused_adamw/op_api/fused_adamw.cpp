/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_adamw.cpp
 * \brief
 */

#include "fused_adamw.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FusedAdamw);

std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensorList*> FusedAdamw(
    const aclTensorList* paramsRef, const aclTensorList* grads, const aclTensorList* expAvgsRef,
    const aclTensorList* expAvgSqsRef, const aclTensorList* maxExpAvgSqsRef, const aclTensorList* stateSteps,
    const aclTensor* gradScaleOptional, const aclTensor* foundInfOptional, float lr, float beta1, float beta2,
    float weightDecay, float eps, bool amsgrad, bool maximize, aclOpExecutor* executor)
{
    L0_DFX(FusedAdamw, paramsRef, grads, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef, stateSteps, gradScaleOptional,
           foundInfOptional, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize);

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(FusedAdamw,
                                                 OP_INPUT(paramsRef, grads, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef,
                                                          stateSteps, gradScaleOptional, foundInfOptional),
                                                 OP_OUTPUT(paramsRef, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef),
                                                 OP_ATTR(lr, beta1, beta2, weightDecay, eps, amsgrad, maximize));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedAdamw ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }
    return std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensorList*>(
        paramsRef, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef);
}

} // namespace l0op
