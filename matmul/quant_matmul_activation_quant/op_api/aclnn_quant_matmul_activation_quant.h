/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_QUANT_MATMUL_ACTIVATION_QUANT_H
#define OP_API_INC_ACLNN_QUANT_MATMUL_ACTIVATION_QUANT_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulActivationQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现QuantMatmulActivationQuant计算
 * @param [in] x1: matmul左矩阵，数据格式支持ND。
 * @param [in] x2: matmul右矩阵，数据格式支持ND。
 * @param [in] x1ScaleOptional: x1的scale。
 * @param [in] x2Scale: x2的scale。
 * @param [in] biasOptional: bias。
 * @param [in] transposeX1: x1是否转置。
 * @param [in] transposeX2: x2是否转置。
 * @param [in] groupSize: 分组大小。
 * @param [in] activationType: 激活函数类型。
 * @param [in] quantMode: 量化模式。
 * @param [in] roundMode: 取整模式。
 * @param [in] scaleAlg: scale算法。
 * @param [in] dstTypeMax: 目标类型最大值。
 * @param [out] yOut: 输出tensor。
 * @param [out] yScaleOut: 输出scale tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
    const aclTensor* biasOptional, bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* yOut,
    aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulActivationQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnQuantMatmulActivationQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulActivationQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_QUANT_MATMUL_ACTIVATION_QUANT_H
