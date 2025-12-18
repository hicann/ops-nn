/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SPARSE_4TO2_QUANT_MATMUL_H
#define OP_API_INC_SPARSE_4TO2_QUANT_MATMUL_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTransSparse4to2Para接口，针对结构化稀疏矩阵weight，压缩为sparseWeight矩阵和index矩阵
 * @domain aclnn_ops_infer
 * 算子功能：将原稀疏的weight压缩为sparseWeight以及相应的index, 用作后续Sparse4to2QuantMatmul的输入
 * @param [in] weight: 矩阵乘对应的右矩阵。
 * @param [in] shape: 矩阵乘的右矩阵对应的shape。
 * @param [out] sparseWeight: 压缩后的weight矩阵。
 * @param [out] sparseWeightDims: 压缩后的weight矩阵shape数组首地址。
 * @param [out] sparseWeightDimsNum: 压缩后的weight矩阵shape数组长度（元素数）。
 * @param [out] index: 压缩后的weight对应的index。
 * @param [out] indexDims: 压缩后的index矩阵shape数组首地址。
 * @param [out] indexDimsNum: 压缩后的index矩阵shape数组长度（元素数）。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnTransSparse4to2Para(
    const int8_t* weight, aclIntArray* shape, int8_t** sparseWeight, int64_t** sparseWeightDims,
    uint64_t* sparseWeightDimsNum, uint8_t** index, int64_t** indexDims, uint64_t* indexDimsNum);

/**
 * @brief aclnnSparse4to2QuantMatmulGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：完成4选2稀疏量化矩阵计算。
 * @param [in] x: matmul左矩阵，数据类型支持：int8。
 * @param [in] sparseWeight: matmul右矩阵，经过aclnnTransSparse4to2Para压缩处理后的稀疏矩阵，数据类型支持：int8。
 * @param [in] index: 压缩后的weight对应的index，数据类型支持：uint8。
 * @param [in] xScale: 量化参数，数据类型支持：float32。
 * @param [in] sparseWeightScale: 量化参数，数据类型支持：float32。
 * @param [in] biasOptional: 偏置，数据类型支持：bfloat16。
 * @param [out] out: 计算结果，数据类型：bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* sparseWeight, const aclTensor* index, const aclTensor* xScale,
    const aclTensor* sparseWeightScale, const aclTensor* biasOptional, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnSparse4to2QuantMatmulGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnSparse4to2QuantMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_SPARSE_4TO2_QUANT_MATMUL_H