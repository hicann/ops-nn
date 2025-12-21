/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_FUSED_MATMUL_H
#define OP_API_INC_FUSED_MATMUL_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现FusedMatmul计算
 * @brief aclnnFusedMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @return aclnnStatus: 返回状态码
 */

ACLNN_API aclnnStatus aclnnFusedMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
    int8_t cubeMathType, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFusedMatmul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnFusedMatmulGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnFusedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FUSED_MATMUL_H