/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CLA_GATE_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_CLA_GATE_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CLA gate 融合反向算子，由上游高精度梯度与前向中间量一次算出
 *        两路 Attention 分支梯度与两路 gate logits 梯度。
 *
 * 计算公式：
 * @li s_g = sigmoid(globalGateLogits), s_l = sigmoid(localGateLogits)
 * @li gradGlobalAttnOut = gradMerged * s_g,  gradLocalAttnOut = gradMerged * s_l
 * @li gradGlobalGateLogitsOut = reduce_sum(gradMerged * globalAttn, axis=head_dim) * s_g * (1 - s_g)
 * @li gradLocalGateLogitsOut  = reduce_sum(gradMerged * localAttn,  axis=head_dim) * s_l * (1 - s_l)
 * 其中 reduce_sum 只沿 head_dim D 归约，不跨 token 或 head。
 *
 * @brief aclnnClaGateBackward 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_ops_infer
 * @param [in] gradMerged: npu device 侧的 aclTensor，输出投影反传到 gate merge 的梯度 G。
 *                         shape [T, N, D]，dtype BFLOAT16/FLOAT16。支持非连续。
 * @param [in] globalAttn: npu device 侧的 aclTensor，前向 Global/CLA 分支 Attention 输出。shape [T, N, D]。
 * @param [in] localAttn:  npu device 侧的 aclTensor，前向 Local/SWA 分支 Attention 输出。shape [T, N, D]。
 * @param [in] globalGateLogits: npu device 侧的 aclTensor，前向 Global gate logits（Sigmoid 前值）。shape [T, N]。
 * @param [in] localGateLogits:  npu device 侧的 aclTensor，前向 Local gate logits（Sigmoid 前值）。shape [T, N]。
 * @param [in] inputAttnLayout: 输入 tensor 排布格式字符串，当前仅支持 "TND"（ND 存储，逻辑 T,N,D）。
 * @param [in] gradGlobalAttnOut: npu device 侧的 aclTensor，Global 分支 Attention 输出梯度。shape [T, N, D]。
 * @param [in] gradLocalAttnOut: npu device 侧的 aclTensor，Local 分支 Attention 输出梯度。shape [T, N, D]。
 * @param [in] gradGlobalGateLogitsOut: npu device 侧的 aclTensor，Global gate logits 梯度输出。shape [T, N]。
 * @param [in] gradLocalGateLogitsOut:  npu device 侧的 aclTensor，Local gate logits 梯度输出。shape [T, N]。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClaGateBackwardGetWorkspaceSize(
    const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
    const aclTensor* globalGateLogits, const aclTensor* localGateLogits, const char* inputAttnLayout,
    const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut, const aclTensor* gradGlobalGateLogitsOut,
    const aclTensor* gradLocalGateLogitsOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnClaGateBackward 的第二段接口，用于执行计算。
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClaGateBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_CLA_GATE_BACKWARD_H_
