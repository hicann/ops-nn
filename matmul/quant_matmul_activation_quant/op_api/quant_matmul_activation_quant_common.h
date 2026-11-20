/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_QUANT_MATMUL_ACTIVATION_QUANT_COMMON_H
#define OP_API_QUANT_MATMUL_ACTIVATION_QUANT_COMMON_H

#include "aclnn/aclnn_base.h"

namespace QBMMActivationQuant {

struct QuantMatmulActivationQuantParams {
    const aclTensor* x1 = nullptr;
    const aclTensor* x2 = nullptr;
    const aclTensor* x1Scale = nullptr;
    const aclTensor* x2Scale = nullptr;
    const aclTensor* bias = nullptr;
    const aclTensor* x3 = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    const char* activationType;
    const char* quantMode;
    const char* roundMode;

    int64_t groupSize;
    int64_t yDtype;
    int64_t scaleAlg;
    double dstTypeMax;
    bool transposeX1;
    bool transposeX2;
};

// Shared validation and executor construction for the original and V2 ACLNN interfaces.
aclnnStatus GetWorkspaceSize(const char* apiName, QuantMatmulActivationQuantParams params, uint64_t* workspaceSize,
                             aclOpExecutor** executor);
aclnnStatus GetWeightNzWorkspaceSize(const char* apiName, QuantMatmulActivationQuantParams params,
                                     uint64_t* workspaceSize, aclOpExecutor** executor);

} // namespace QBMMActivationQuant

#endif // OP_API_QUANT_MATMUL_ACTIVATION_QUANT_COMMON_H
