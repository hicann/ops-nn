/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H
#define OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaLayerNormQuantGetWorkspaceSize
 */
ACLNN_API aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* smoothScalesOptional, double epsilon, const char* quantMode,
    aclTensor* out, aclTensor* quantScale, aclTensor* quantOffsetOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/*
 * @brief aclnnAdaLayerNormQuant
 */
ACLNN_API aclnnStatus
aclnnAdaLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H