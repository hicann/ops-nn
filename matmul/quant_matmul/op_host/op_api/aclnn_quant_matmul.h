/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_MM_H_
#define OP_API_INC_QUANT_MM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmulGetWorkspaceSize is deprecated, use aclnnQuantMatmulV4GetWorkspaceSize instead")
ACLNN_API aclnnStatus aclnnQuantMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, float deqScale, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmul的第二段接口，用于执行计算。
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmul is deprecated, use aclnnQuantMatmulV4 instead")
ACLNN_API aclnnStatus
aclnnQuantMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnQuantMatmulV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACL_DEPRECATED_MESSAGE(
    "aclnnQuantMatmulV2GetWorkspaceSize is deprecated, use aclnnQuantMatmulV4GetWorkspaceSize instead")
ACLNN_API aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, bool adjX1, bool adjX2,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulV2的第二段接口，用于执行计算。
 */
ACL_DEPRECATED_MESSAGE("aclnnQuantMatmulV2 is deprecated, use aclnnQuantMatmulV4 instead")
ACLNN_API aclnnStatus
aclnnQuantMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_MM_H_
