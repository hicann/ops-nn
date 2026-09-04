/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_NZ_H
#define OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_NZ_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulActivationQuantWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现QuantMatmulActivationQuantWeightNz计算
 * @param [in] x1: matmul左矩阵，数据类型支持：FLOAT8_E4M3FN、FLOAT8_E5M2数据类型，数据格式支持ND。
 * @param [in] x2: matmul右矩阵，数据类型支持：FLOAT8_E4M3FN数据类型，数据格式支持NZ。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
    const aclTensor* biasOptional, bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* yOut,
    aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulActivatiionQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnQuantMatmulActivatiionQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulActivationQuantWeightNz(void* workspace, uint64_t workspaceSize,
                                                              aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // QuantMatmulActivationQuantWeightNz
