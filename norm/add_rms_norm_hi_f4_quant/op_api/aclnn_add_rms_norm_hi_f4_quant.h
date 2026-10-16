/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ADD_RMS_NORM_HI_F4_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_RMS_NORM_HI_F4_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAddRmsNormHiF4QuantGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                               const aclTensor* gamma, const aclTensor* beta,
                                                               double epsilon, bool outputRstd, int64_t scaleAlg,
                                                               aclTensor* yOut, aclTensor* xOut, aclTensor* scaleOut,
                                                               aclTensor* rstdOut, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnAddRmsNormHiF4Quant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
