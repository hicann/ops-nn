/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
#define OP_API_INC_ACLNN_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate the workspace required to check and unscale the gradients in place.
 * @domain aclnnop_math
 * @param [in,out] scaledGrads  Gradient tensor list to update.
 * @param [in,out] foundInf     Single-element FLOAT tensor; set to one on non-finite input, otherwise preserved.
 * @param [in] invScale         Read-only single-element FLOAT tensor containing the inverse scale.
 * @param [out] workspaceSize   Required device workspace size.
 * @param [out] executor        Executor containing the computation.
 * @return aclnnStatus          Status code.
 */
ACLNN_API aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList* scaledGrads,
                                                                           const aclTensor* foundInf,
                                                                           const aclTensor* invScale,
                                                                           uint64_t* workspaceSize,
                                                                           aclOpExecutor** executor);

/**
 * @brief Execute the in-place non-finite check and gradient unscale operation.
 */
ACLNN_API aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_H_
