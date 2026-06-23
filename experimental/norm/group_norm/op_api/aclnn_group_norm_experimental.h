/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPERIMENTAL_NORM_ACLNN_GROUP_NORM_EXPERIMENTAL_H_
#define EXPERIMENTAL_NORM_ACLNN_GROUP_NORM_EXPERIMENTAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnGroupNormGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gammaOptional, const aclTensor* betaOptional, int64_t n, int64_t c,
    int64_t hxw, int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnGroupNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // EXPERIMENTAL_NORM_ACLNN_GROUP_NORM_EXPERIMENTAL_H_
