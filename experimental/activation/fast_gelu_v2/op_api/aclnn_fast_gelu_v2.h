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
 * @file aclnn_fast_gelu_v2.h
 * @brief ACLNN L2 API declaration for FastGeluV2
 */

#ifndef ACLNN_FAST_GELU_V2_H_
#define ACLNN_FAST_GELU_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size for aclnnFastGeluV2
 * @param x [in] Input tensor
 * @param out [in] Output tensor
 * @param workspaceSize [out] Required workspace size in bytes
 * @param executor [out] Executor handle
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnFastGeluV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute FastGeluV2 computation
 * @param workspace [in] Workspace memory address
 * @param workspaceSize [in] Workspace size in bytes
 * @param executor [in] Executor handle
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnFastGeluV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FAST_GELU_V2_H_
