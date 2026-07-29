/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_FUSED_MATMUL_GELU_H_
#define OP_API_INC_ACLNN_FUSED_MATMUL_GELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedMatmulGelu first-stage API. Calculates workspace size and builds executor.
 *
 * Computes:
 *   y = gelu(x @ weight^T + bias)
 *
 * @param [in] x: NPU tensor, FLOAT16/BFLOAT16, ND, shape [..., K].
 * @param [in] weight: NPU tensor, FLOAT16/BFLOAT16, ND, shape [N, K].
 * @param [in] bias: Optional NPU tensor, FLOAT16/BFLOAT16, ND, shape [N]. Can be nullptr.
 * @param [in] approximate: Host int64. 1 means tanh approximate mode.
 * @param [out] y: NPU tensor, FLOAT16/BFLOAT16, ND, shape [..., N].
 * @param [out] workspaceSize: Workspace size in bytes.
 * @param [out] executor: Op executor.
 * @return aclnnStatus.
 */
ACLNN_API aclnnStatus aclnnFusedMatmulGeluGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                           const aclTensor* bias, int64_t approximate, aclTensor* y,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFusedMatmulGelu second-stage API. Runs executor.
 *
 * @param [in] workspace: Workspace address.
 * @param [in] workspaceSize: Workspace size.
 * @param [in] executor: Op executor.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus.
 */
ACLNN_API aclnnStatus aclnnFusedMatmulGelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_FUSED_MATMUL_GELU_H_
