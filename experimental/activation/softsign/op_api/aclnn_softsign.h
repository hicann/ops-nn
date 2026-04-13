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
  * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_softsign.h
 * @brief ACLNN L2 API declaration for Softsign operator
 *
 * Two-phase API:
 *   - aclnnSoftsignGetWorkspaceSize: compute workspace size, create executor
 *   - aclnnSoftsign: execute computation
 */

#ifndef ACLNN_SOFTSIGN_H_
#define ACLNN_SOFTSIGN_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size for aclnnSoftsign
 * @param self [in] Input tensor
 * @param out [in] Output tensor
 * @param workspaceSize [out] Required workspace size
 * @param executor [out] Executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnSoftsignGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute Softsign computation
 * @param workspace [in] Workspace memory address
 * @param workspaceSize [in] Workspace size
 * @param executor [in] Executor
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnSoftsign(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SOFTSIGN_H_
