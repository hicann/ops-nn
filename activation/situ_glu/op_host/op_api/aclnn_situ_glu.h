/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_SITU_GL_H_
#define ACLNN_SITU_GL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSituGlu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：SituGlu是一个门控线性单元函数，它将输入张量沿着指定的维度dim平均分成两个张量，
 * 对其中一块做beta门控变换后与另一块逐元素相乘。
 *
 * 计算公式：
 * $$ situ\_a = beta \cdot tanh(g/beta) / (1 + e^{-g}) $$
 * $$ out = situ\_a \cdot u $$
 * 其中，g和u表示x根据指定dim均分后的两块张量，activateLeft为true时g取前半部分、u取后半部分，
 * 否则g取后半部分、u取前半部分；当linearBeta大于0时，u = linearBeta \cdot tanh(u / linearBeta)。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *    A[(x)] -->B([l0op::Contiguous])
 *    B -->C([l0op::SituGlu])
 *    D((dim)) -->C
 *    E((beta)) -->C
 *    F((linearBeta)) -->C
 *    G((activateLeft)) -->C
 *    C -->H([l0op::ViewCopy])
 *    H -->I[(out)]
 * ```
 *
 * @param [in] x: 数据类型支持FLOAT、FLOAT16、BFLOAT16数据类型，tensor的维度必须大于0，
 * 且shape必须在入参dim对应的维度上可以整除2，shape表示为$(*_1,N,*_2)$其中$*$表示任何数量的附加维，
 * $N$表示dim指定的维度大小，支持非连续的Tensor，数据格式支持ND。
 * @param [in] dim: 表示要对输入x进行门控计算的维度，数据类型支持INT64，
 * 取值范围[-x.dim，x.dim-1]，默认值为-1。
 * @param [in] beta: 数据类型支持DOUBLE，取值须为非0的有限值，默认值为1.0。
 * @param [in] linearBeta: 数据类型支持DOUBLE，取值须为有限值，默认值为0.0，大于0时对u进行变换。
 * @param [in] activateLeft: 计算属性，host侧的布尔值，表示门控计算数据块的方向，
 * 默认值为true，表示对左边数据块做门控计算。
 * @param [out] out: 数据类型支持FLOAT、FLOAT16、BFLOAT16数据类型且数据类型必须与x一致，
 * shape为$(*_1,M,*_2)$其中$*$表示x中对应维度，$M = N/2$，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSituGluGetWorkspaceSize(const aclTensor* x, int64_t dim, double beta, double linearBeta,
                                                   bool activateLeft, const aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnSituGlu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSituGluGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSituGlu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SITU_GL_H_
