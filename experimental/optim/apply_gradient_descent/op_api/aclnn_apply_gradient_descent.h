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
 * @file aclnn_apply_gradient_descent.h
 * @brief ACLNN L2 API declaration for ApplyGradientDescent
 *
 * Inplace update: var = var - alpha * delta.
 *   - var:   aclTensor* (mutated inplace; both input and output)
 *   - alpha: scalar tensor (1 element), same dtype as var
 *   - delta: same shape & dtype as var
 *
 * Two-stage interface:
 * - aclnnApplyGradientDescentGetWorkspaceSize: compute workspace size, create executor
 * - aclnnApplyGradientDescent: execute computation
 */

#ifndef ACLNN_APPLY_GRADIENT_DESCENT_H_
#define ACLNN_APPLY_GRADIENT_DESCENT_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnApplyGradientDescentGetWorkspaceSize(aclTensor* var, const aclTensor* alpha,
                                                                const aclTensor* delta, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnApplyGradientDescent(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_APPLY_GRADIENT_DESCENT_H_
