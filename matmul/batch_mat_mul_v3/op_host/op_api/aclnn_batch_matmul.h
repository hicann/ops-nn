/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_BATCHMATMUL_H
#define OP_API_ACLNN_BATCHMATMUL_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBatchMatMul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnBatchMatMulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnBatchMatMul的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnBatchMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnBatchMatMulWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：相对于aclnnBatchMatMul, mat2为NZ格式。
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16类型，且需要与mat2的数据类型需满足数据类型推导规则，
 * shape需要与mat2满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] mat2: npu
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16类型，且数据类型需要与self的数据类型需满足数据类型推导规则,
 * shape需要与self满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持NZ。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16类型，且数据类型需要与self保持一致，shape要求与self@mat2的后两维保持一致。
 * 支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] cubeMathType:
 * INT8类型的枚举值，用于判断Cube单元应该使用那种计算逻辑进行运算，可通过此开关使能如HFLOAT32等功能
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnBatchMatMulWeightNz的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBatchMatMulWeightNz获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBatchMatMulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_BATCHMATMUL_H