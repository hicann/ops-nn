/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_LOGIT_GRAD_H_
#define OP_API_INC_ACLNN_LOGIT_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the workspace size and executor for LogitGrad.
 * @param x Forward input tensor, supporting non-contiguous views in ND format.
 * @param dy Output gradient with the same logical shape and dtype as x.
 * @param eps Epsilon attribute of the original LogitGrad operator.
 * @param out Input gradient with the same logical shape and dtype as x, supporting non-contiguous views.
 * @param workspaceSize Required device workspace size.
 * @param executor Executor containing the computation and any required layout conversions.
 * @return aclnnStatus Status code.
 */
ACLNN_API aclnnStatus aclnnLogitGradGetWorkspaceSize(const aclTensor* x, const aclTensor* dy, double eps,
                                                     const aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief Execute LogitGrad using the workspace and executor returned by the first phase.
 */
ACLNN_API aclnnStatus aclnnLogitGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_LOGIT_GRAD_H_
