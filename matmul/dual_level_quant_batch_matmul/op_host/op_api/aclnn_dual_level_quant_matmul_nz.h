/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_DUAL_LEVEL_QUANT_MATMUL_NZ_H_
#define OP_API_ACLNN_DUAL_LEVEL_QUANT_MATMUL_NZ_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDualLevelQuantMatmulWeightNz 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现二级量化 MxFP4@MxFP4 场景的右矩阵为NZ场景的矩阵乘。
 * @param [in] x1: matmul左矩阵，数据格式支持ND，数据类型支持：float4_e2m1。
 * @param [in] x2: matmul右矩阵，数据格式支持NZ，数据类型支持：float4_e2m1。
 * @param [in] bias: 偏置，数据类型支持：float。
 * @param [in] x1Level0Scale: 左矩阵一级反量化scale参数，数据类型支持：float。
 * @param [in] x2Level0Scale: 右矩阵一级反量化scale参数，数据类型支持：float。
 * @param [in] x1Level1Scale: 左矩阵二级反量化scale参数，数据类型支持：float8_e8m0。
 * @param [in] x2Level1Scale: 右矩阵二级反量化scale参数，数据类型支持：float8_e8m0。
 * @param [in] level0GroupSize:
 * 一级量化下，对输入x1和x2进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。
 * @param [in] level1GroupSize:
 * 二级量化下，对输入x1和x2进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。
 * @param [in] y: 计算结果，数据类型支持：float16, bfloat16。
 * @param [in] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [in] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Level0Scale, const aclTensor* x2Level0Scale,
    const aclTensor* x1Level1Scale, const aclTensor* x2Level1Scale, const aclTensor* optionalBias, bool transposeX1,
    bool transposeX2, int64_t level0GroupSize, int64_t level1GroupSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDualLevelQuantMatmulWeightNz 的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnWeightQuantBatchMatmulNzGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnDualLevelQuantMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_DUAL_LEVEL_QUANT_MATMUL_NZ_H_