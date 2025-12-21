/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ADDBMM_H_
#define OP_API_INC_ADDBMM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddbmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：首先进行batch1、batch2的矩阵乘计算，然后将该结果按照第一维（batch维度）批处理相加，将三维向量
 * 压缩为二维向量（shape大小为后两维的shape），然后该结果与α作乘积计算，再与β和self的乘积求和得到结果
 * 计算公式：
 * $$ out = βself+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且需要与batch1@batch2的数据类型需满足数据类型推导规则，
 * shape需要与batch1@batch2的后两维满足broadcast关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] batch1: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch2的数据类型需满足数据类型推导规则,
 * shape需要与batch2满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] batch2: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch1的数据类型需满足数据类型推导规则
 * shape需要与batch1满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] beta: host侧的aclScalar，数据类型需要可转换成self与batch1@batch2推导后的数据类型。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与batch1@batch2推导后的数据类型。
 * @param [in] cubeMathType:
 * INT8类型的枚举值，用于判断Cube单元应该使用那种计算逻辑进行运算，可通过此开关使能如HFLOAT32等功能
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与self保持一致，shape要求与batch1@batch2的后两维保持一致。
 * 支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddbmm的第二段接口，用于执行计算。
 *
 * 算子功能：首先进行batch1、batch2的矩阵乘计算，然后将该结果按照第一维（batch维度）批处理相加，将三维向量
 * 压缩为二维向量（shape大小为后两维的shape），然后该结果与α作乘积计算，再与β和self的乘积求和得到结果
 * 计算公式：
 * $$ out = βself+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddbmmGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAddbmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：首先进行batch1、batch2的矩阵乘计算，然后将该结果按照第一维（batch维度）批处理相加，将三维向量
 * 压缩为二维向量（shape大小为后两维的shape），然后该结果与α作乘积计算，再与β和selfRef的乘积求和得到结果
 * 计算公式：
 * $$ selfRef = βselfRef+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且需要与batch1@batch2的数据类型需满足数据类型推导规则，
 * shape需要与batch1@batch2的后两维满足broadcast关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] batch1: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch2的数据类型需满足数据类型推导规则,
 * shape需要与batch2满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] batch2: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16类型，且数据类型需要与batch1的数据类型需满足数据类型推导规则
 * shape需要与batch1满足bmm输入约束关系。支持非连续的Tensor，支持空Tensor传入，数据格式支持ND。
 * @param [in] beta: host侧的aclScalar，数据类型需要可转换成self与batch1@batch2推导后的数据类型。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与batch1@batch2推导后的数据类型。
 * @param [in] cubeMathType:
 * INT8类型的枚举值，用于判断Cube单元应该使用那种计算逻辑进行运算，可通过此开关使能如HFLOAT32等功能
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAddbmm的第二段接口，用于执行计算。
 *
 * 算子功能：首先进行batch1、batch2的矩阵乘计算，然后将该结果按照第一维（batch维度）批处理相加，将三维向量
 * 压缩为二维向量（shape大小为后两维的shape），然后该结果与α作乘积计算，再与β和selfRef的乘积求和得到结果
 * 计算公式：
 * $$ selfRef = βselfRef+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAddbmmGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceAddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BADDBMM_H_
