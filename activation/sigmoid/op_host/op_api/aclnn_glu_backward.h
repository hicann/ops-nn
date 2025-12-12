/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_GLU_BACKWARD_H_
#define OP_API_INC_GLU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：GLU的反向。
 *
 * $$
 * \frac{\partial GLU(a,b)}{\partial(a,b)}=cat(\sigma(b),\sigma(b) \otimes a \otimes (1-\sigma(b)))
 * $$
 *
 * 数学计算表达式：
 * 假设输出的GLUGrad有两部分组成:out=[a_grad, b_grad]，则：
 * sig_b = sigmoid(b)
 * **a_grad** = y_grad * sig_b
 * **b_grad** = a_grad * (a - a * sig_b)
 * 其中：y_grad 为gradOut，a表示的是输入张量根据指定dim进行均分后的前部分张量，b表示后半部分张量。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *    A0[(gradOut)] -->B0([l0op::Contiguous])-->C1([l0op::Mul])-->C2([l0op::Mul])
 *    A1[(self)] -->B1([l0op::Contiguous])
 *    B1 -->D0([l0op::SplitV])--a--> C0-->G0([l0op::Sub])
 *    D0--a-->G0-->C2--b_grad-->H0([l0op::ConcatD])
 *    E0((dim)) -->D0--b-->D1([l0op::Sigmoid])-->C0([l0op::Mul])
 *    D1-->C1--a_grad-->H0
 *    E0-->H0
 *    H0 -->F0([l0op::ViewCopy])--> J0[(out)]
 * ```
 *
 * @param [in] gradOut: 表示梯度更新系数，数据类型支持DOUBLE,FLOAT,FLOAT16数据类型，数据类型必须与self的数据类型一致，
 * shape为$(*_1,M,*_2)$其中$*$表示self中对应维度，$M = N /2$，支持非连续的Tensor，数据格式支持ND。
 * @param [in] self:
 * 数据类型支持DOUBLE,FLOAT,FLOAT16数据类型，tensor的维度必须大于0，且shape必须在入参dim对应的维度上可以整除2，
 * shape表示为$(*_1,N,*_2)$其中$*$表示任何数量的附加维，$N$表示dim指定的维度大小，支持非连续的Tensor，数据格式支持ND。
 * @param [in] dim: 表示要拆分输入self的维度，数据类型支持INT64，取值范围[-self.dim，self.dim-1]。
 * @param [out] out: 数据类型支持DOUBLE,FLOAT,FLOAT16数据类型，数据类型必须与self的数据类型一致，
 * shape必须与self的shape一致，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGluBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* self, int64_t dim,
                                                       const aclTensor* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnGluBackward的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGluBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GLU_BACKWARD_H_
