/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CTCLOSS_H_
#define OP_API_INC_CTCLOSS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCtcLoss的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 计算连接时序分类损失值。
 *
 * 定义$y_{k}^{t}$表示在时刻$t$时真实字符为$k$的概率。(一般地，$y_{k}^{t}$是经过softmax之后的输出矩阵中的一个元素)。
 * 将字符集$L^{'}$可以构成的所有序列的集合称为$L^{'T}$，将$L^{'T}$中的任意一个序列称为路径，并标记为$π$。$π$的分布为公式(1)：
 *
 * $$
 * p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
 * $$
 *
 * 定义多对一(many to one)映射B: $L^{'T} \to L^{\leq T}$，通过映射B计算得到$l \in L^{\leq T}$的条件概率，
 * 等于对应于$l$的所有可能路径的概率之和，公式(2):
 *
 * $$
 * p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
 * $$
 *
 * 将找到使$p(l|x)$值最大的$l$的路径的任务称为解码，公式(3)：
 *
 * $$
 * h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
 * $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 * A[(logProbs)] -->B([l0op::Contiguous])-->D([l0op::CTCLossV2])
 * A1[(targets)] -->B1([l0op::Contiguous])-->D
 * A2[(targetLengths)] -->B2([ConvertToTensor])-->D
 * A3[(inputLengths)] -->B3([ConvertToTensor])-->D
 * A4((blank)) -->D
 * A6((zeroInfinity)) -->D
 * D--negLogLikelihood--> F1([l0op::ViewCopy])--> J[(negLogLikelihoodOut)]
 * D--logAlpha-->H([l0op::ViewCopy])-->J1[(logAlphaOut)]
 * ```
 *
 * @param [in] logProbs(aclTensor*): 数据类型支持FLOAT,DOUBLE数据类型，shape为($T,N,C$)，
 * $T$为输入长度，$N$为批处理大小，$C$为类别数，必须大于0，包括空白标识，该Tensor表示输出的对数概率，
 * 支持[非连续的Tensor](https://)，数据格式支持ND。
 * @param [in] targets(aclTensor*): 数据类型支持INT64,INT32,BOOL,FLOAT,FLOAT16数据类型，当shape为($N,S$)，
 * $S$为不小于$targetLengths$中的最大值的值；或者shape为(SUM($targetLengths$))，假设$targets$是未填充的而且在1维内级联的；
 * 支持[非连续的Tensor](https://)，数据格式支持ND。
 * @param [in] inputLengths(aclIntArray*)：数据类型支持UINT8,INT8,INT16,INT32,INT64，数组长度为$N$，
 * 数组中的每个值必须小于等于$T$。
 * @param [in] targetLengths(aclIntArray*)：数据类型支持UINT8,INT8,INT16,INT32,INT64，数组长度为$N$，
 * 当targets的shape为($N,S$)时，数组中的每个值必须小于等于$S$。
 * @param [in] blank(int)：int整型，空白标识，默认为0，数值必须小于$C$大于等于0。
 * @param [in] zeroInfinity(bool)：bool类型，表示是否将无限损耗和相关梯度归零，默认值为$False$。
 * @param [out] negLogLikelihoodOut(aclTensor*): 输出的损失值，数据类型FLOAT,DOUBLE；
 * 输出一个大小为($N$)的Tensor，支持[非连续的Tensor](https://)，数据格式支持ND。
 * @param [out] logAlphaOut(aclTensor*): 数据类型支持FLOAT,DOUBLE数据类型(数据类型必须和logProbs一致)，
 * 表示输入到目标的可能跟踪的概率，该Tensor为3维，支持[非连续的Tensor](https://)，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCtcLossGetWorkspaceSize(
    const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths,
    const aclIntArray* targetlengths, int64_t blank, bool zeroInfinity, aclTensor* negLogLikelihoodOut,
    aclTensor* logAlphaOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnCtcLoss 的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCtcLossGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnCtcLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_CTCLOSS_H_
