/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GEMM_SYRK_H
#define OP_API_INC_GEMM_SYRK_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现对称秩k更新（syrk）计算：C = alpha * (A @ A^T) + beta * C，
 * 其中C为对称矩阵，输入输出同地址（原地更新），算子内部输出完整矩阵结果。
 * transposeX为true时，a以转置的(k, m)存储，计算C = alpha * (A^T @ A) + beta * C。
 * @brief aclnnGemmSyrk的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param a [in] 输入矩阵A，shape为(m, k)或(batch, m, k)；transposeX为true时为(k, m)或(batch, k,
 * m)，数据类型float16/bfloat16。
 * @param cRef [in/out] 对称矩阵C，shape为(m, m)或(batch, m, m)，原地读入beta*C并写回结果。
 * @param alphaOptional [in] 缩放标量alpha，nullptr时默认为1.0。
 * @param betaOptional [in] 缩放标量beta，nullptr时默认为1.0。
 * @param transposeX [in] 是否按转置布局解读a，默认语义为false（调用方直接传false/true）。
 * @param fillMode [in] 输出区域模式，支持"full"（完整对称矩阵）/"up"（上三角）/"low"（下三角）；
 * 当前仅支持"full"，传"up"/"low"返回参数错误；传nullptr时默认"full"。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGemmSyrkGetWorkspaceSize(const aclTensor* a, aclTensor* cRef, const aclScalar* alphaOptional,
                                                    const aclScalar* betaOptional, bool transposeX,
                                                    const char* fillMode, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief aclnnGemmSyrk的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存地址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGemmSyrkGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGemmSyrk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GEMM_SYRK_H
