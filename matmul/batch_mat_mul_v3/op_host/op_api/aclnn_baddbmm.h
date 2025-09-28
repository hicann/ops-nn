/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BADDBMM_H_
#define OP_API_INC_BADDBMM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBaddbmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成加法计算
 * 计算公式：计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且需要与batch1@batch2保持一致；shape需要与batch1@batch2满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] batch1: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch2保持一致，shape需要与batch2满足bmm输入约束关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] batch2: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch1保持一致，shape需要与batch1满足bmm输入约束关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta: host侧的aclScalar，默认为1
 * @param [in] alpha: host侧的aclScalar，默认为1
 * @param [in] cubeMathType:
 * INT8类型的枚举值，用于判断Cube单元应该使用那种计算逻辑进行运算，可通过此开关使能如HFLOAT32等功能
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，dtype和format均需要与self、batch1@batch2保持一致。
 * 支持非连续的Tensor，数据格式支持ND。shape不做限制，可传入任意shape，不要求与self、batch1@batch2的shape保持一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBaddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBaddbmm的第二段接口，用于执行计算。
 *
 * 算子功能：完成加法计算
 * 计算公式：计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBaddbmmGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceBaddbmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成加法计算
 * 计算公式：计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且需要与batch1@batch2保持一致；shape需要与batch1@batch2满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] batch1: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch2保持一致，shape需要与batch2满足bmm输入约束关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] batch2: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch1保持一致，shape需要与batch1满足bmm输入约束关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] beta: host侧的aclScalar，默认为1
 * @param [in] alpha: host侧的aclScalar，默认为1
 * @param [in] cubeMathType:
 * INT8类型的枚举值，用于判断Cube单元应该使用那种计算逻辑进行运算，可通过此开关使能如HFLOAT32等功能
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBaddbmm的第二段接口，用于执行计算。
 *
 * 算子功能：完成加法计算
 * 计算公式：计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceBaddbmmGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BADDBMM_H_
