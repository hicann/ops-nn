/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL
#define OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnTransposeQuantBatchMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现TransposeQuantBatchMatmul计算。
 * @param [in] x1: matmul左矩阵，数据类型支持：float8_e4m3fn, float8_e5m2。
 * @param [in] x2: matmul右矩阵，数据类型支持：float8_e4m3fn, float8_e5m2。
 * @param [in] bias: 偏置，当前不支持。
 * @param [in] x1Scale: 量化参数中的缩放因子，数据类型支持：float32。
 * @param [in] x2Scale: 量化参数中的缩放因子，数据类型支持：float32。
 * @param [in] dtype: 输出矩阵的数据类型。数据类型支持：int32。
 * @param [in] groupSize: 输出矩阵的分组大小。数据类型支持：int32。
 * @param [in] permX1: 表示输入x1的转置数组。
 * @param [in] permX2: 表示输入x2的转置数组。
 * @param [in] permY: 表示输入y的转置数组。
 * @param [in] batchSplitFactor: 是否重新拆分shape。数据类型支持：int32。
 * @param [out] out: 计算结果，数据类型：float16, bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const int32_t dtype, const int32_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2,
    const aclIntArray* permY, const int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnTransposeBatchMatmul的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMul(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL
