/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GLU_H_
#define OP_API_INC_GLU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGlu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：GLU是一个门控线性单元函数，它将输入张量沿着指定的维度dim平均分成两个张量，
 * 并将其前部分张量与后部分张量的Sigmoid函数输出的结果逐元素相乘.
 *
 * $$
 * GLU(a,b)=a \otimes \sigma(b)
 * $$
 *
 * a表示的是输入张量根据指定dim进行均分后的前部分张量，b表示后半部分张量。
 *
 * 计算图：
 * 1、当self的dtype等于out的dtype时：
 * ```mermaid
 * graph LR
 *    A[(self)] -->B([l0op::Contiguous])
 *    B -->D([l0op::SplitV])--a--> C3
 *    E((dim)) -->D--b-->D1([l0op::Sigmoid])-->C3([l0op::Mul])
 *    C3 -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * 2、当self的dtype不等于out的dtype时：
 * ```mermaid
 * graph LR
 *    A[(self)] -->B([l0op::Contiguous])
 *    B -->D([l0op::SplitV])--a--> C3
 *    E((dim)) -->D--b-->D1([l0op::Sigmoid])-->C3([l0op::Mul])
 *    C3 -->F0([l0op::Cast])-->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: 数据类型支持DOUBLE,FLOAT,BFLOAT16,FLOAT16数据类型，tensor的维度必须大于0，
 * 且shape必须在入参dim对应的维度上可以整除2，shape表示为$(*_1,N,*_2)$其中$*$表示任何数量的附加维，$N$表示dim指定的维度大小，
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] dim: 表示要拆分输入self的维度，数据类型支持INT64，取值范围[-self.dim，self.dim-1]。
 * @param [out] out: 数据类型支持DOUBLE,FLOAT,BFLOAT16,FLOAT16数据类型，数据类型必须可以由self cast得到，
 * shape为$(*_1,M,*_2)$其中$*$表示self中对应维度，$M = N /2$，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* out,
                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGlu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGtTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGlu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GLU_H_
