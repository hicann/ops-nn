/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_GELU_MUL_H_
#define OP_API_INC_ACLNN_GELU_MUL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the workspace size and executor for GeluMul.
 * @param input Input tensor, supporting non-contiguous views in ND format.
 * @param approximateOptional Optional activation mode, "none" or "tanh"; nullptr is treated as "none".
 * @param out Output tensor whose shape keeps the input dims except the last one, which is halved;
 *            supporting non-contiguous views in ND format.
 * @param workspaceSize Required device workspace size.
 * @param executor Executor containing the computation and any required layout conversions.
 * @return aclnnStatus Status code.
 */
ACLNN_API aclnnStatus aclnnGeluMulGetWorkspaceSize(const aclTensor* input, char* approximateOptional,
                                                   const aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief Execute GeluMul using the workspace and executor returned by the first phase.
 */
ACLNN_API aclnnStatus aclnnGeluMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_GELU_MUL_H_
