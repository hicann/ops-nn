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
 * @file apply_gradient_descent.h
 * @brief ACLNN L0 API declaration for ApplyGradientDescent
 *
 * Computes var_out = var - alpha * delta, elementwise (inplace at the L2 layer).
 */

#ifndef OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_
#define OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_

#include "opdev/op_executor.h"

namespace l0op {

// Returns the computed var_out tensor (allocated by the executor). The L2 API
// completes the inplace write-back to the caller's var buffer via ViewCopy.
const aclTensor* ApplyGradientDescent(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta,
                                      aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_
