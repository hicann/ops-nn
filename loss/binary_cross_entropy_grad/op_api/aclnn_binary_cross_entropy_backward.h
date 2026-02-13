/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_BINARY_CROSS_ENTROPY_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_BINARY_CROSS_ENTROPY_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBinaryCrossEntropyBackwardGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小
 * @domain aclnn_ops_train
 * 算子功能：求二元交叉熵反向传播的梯度值
 */
ACLNN_API aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weightOptional,
    int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/*
 * @brief aclnnBinaryCrossEntropyBackward的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnBinaryCrossEntropyBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BINARY_CROSS_ENTROPY_BACKWARD_H_
