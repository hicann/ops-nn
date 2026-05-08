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
 * @file aclnn_hardsigmoid.h
 * @brief ACLNN L2 API - aclnnHardsigmoid / aclnnInplaceHardsigmoid
 */

#ifndef ACLNN_HARDSIGMOID_H_
#define ACLNN_HARDSIGMOID_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardsigmoid first stage: compute workspace size
 * @param self [in] input tensor (FLOAT, FLOAT16, BF16, INT32)
 * @param out  [in] output tensor (same dtype as self)
 * @param workspaceSize [out] required workspace size
 * @param executor [out] op executor
 */
ACLNN_API aclnnStatus aclnnHardsigmoidGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHardsigmoid second stage: execute
 */
ACLNN_API aclnnStatus
aclnnHardsigmoid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceHardsigmoid first stage: compute workspace size (inplace)
 * @param self [in] input tensor (also used as output)
 */
ACLNN_API aclnnStatus
aclnnInplaceHardsigmoidGetWorkspaceSize(const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceHardsigmoid second stage: execute (inplace)
 */
ACLNN_API aclnnStatus
aclnnInplaceHardsigmoid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_HARDSIGMOID_H_
