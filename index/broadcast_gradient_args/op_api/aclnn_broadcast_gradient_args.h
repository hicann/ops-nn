/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_BROADCAST_GRADIENT_ARGS_H_
#define OP_API_INC_BROADCAST_GRADIENT_ARGS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBroadcastGradientArgs的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：在反向传播过程中，根据两个张量在正向传播时的原始形状，自动识别出它们因广播机制而扩展的维度，
 *           并输出需要在哪些维度上对梯度进行约简，以便将梯度从广播后的形状还原为每个原始张量的形状。
 *
 * @param [in] x1: npu device侧的aclTensor，数据类型支持INT32、INT64，shape必须为1维，
 *                 data是原始张量a的shape。支持非连续的Tensor，数据格式支持ND。
 * @param [in] x2: npu device侧的aclTensor，数据类型与x1一致，shape必须为1维，
 *                 data是原始张量b的shape。支持非连续的Tensor，数据格式支持ND。
 * @param [out] y1: npu device侧的aclTensor，表示x1对应的张量shape中需要广播的索引，
 *                 数据类型与x1一致，shape必须为1维。数据格式支持ND。
 * @param [out] y2: npu device侧的aclTensor，表示x2对应的张量shape中需要广播的索引，
 *                 数据类型与x1一致，shape必须为1维。数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBroadcastGradientArgsGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                                 aclTensor* y1, aclTensor* y2, uint64_t* workspaceSize,
                                                                 aclOpExecutor** executor);

/**
 * @brief aclnnBroadcastGradientArgs的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBroadcastGradientArgsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBroadcastGradientArgs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BROADCAST_GRADIENT_ARGS_H_
