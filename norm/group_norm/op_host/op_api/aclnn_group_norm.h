/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUP_NORM_H_
#define OP_API_INC_GROUP_NORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupNorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnGroupNormGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGroupNorm的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnGroupNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GROUP_NORM_H_
