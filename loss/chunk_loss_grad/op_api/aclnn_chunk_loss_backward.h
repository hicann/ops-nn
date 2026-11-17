/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "aclnn/aclnn_base.h"
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
// alpha is the reciprocal of the forward normalization constant.
ACLNN_API aclnnStatus aclnnChunkLossBackwardGetWorkspaceSize(const aclTensor* hidden, const aclTensor* weight,
                                                             const aclTensor* labels, double alpha,
                                                             const aclTensor* softmaxMax, const aclTensor* softmaxSum,
                                                             const aclTensor* gradOutput, int64_t ignoreIndex,
                                                             aclTensor* gradWeightOut, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);
ACLNN_API aclnnStatus aclnnChunkLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);
#ifdef __cplusplus
}
#endif
