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
 * @file aclnn_fused_bias_leaky_relu.h
 * @brief ACLNN L2 API declaration for FusedBiasLeakyRelu
 *
 * Two-phase ACLNN interface:
 * - GetWorkspaceSize: compute workspace size, create executor
 * - aclnnFusedBiasLeakyRelu: execute computation
 */

#ifndef ACLNN_FUSED_BIAS_LEAKY_RELU_H_
#define ACLNN_FUSED_BIAS_LEAKY_RELU_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size for aclnnFusedBiasLeakyRelu
 * @param x [in] Input tensor
 * @param bias [in] Bias tensor (same shape as x)
 * @param negativeSlope [in] LeakyReLU negative slope
 * @param scale [in] Output scale factor
 * @param out [in] Output tensor
 * @param workspaceSize [out] Required workspace size
 * @param executor [out] Executor handle
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnFusedBiasLeakyReluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    double negativeSlope,
    double scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute FusedBiasLeakyRelu computation
 * @param workspace [in] Workspace memory
 * @param workspaceSize [in] Workspace size
 * @param executor [in] Executor handle
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnFusedBiasLeakyRelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FUSED_BIAS_LEAKY_RELU_H_
