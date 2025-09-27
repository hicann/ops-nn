/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEAKY_RELU_BACKWARD_H_
#define OP_API_INC_LEAKY_RELU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnLeakyReluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：激活函数反向。
 *
 * @param [in] gradOutput: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE，且数据类型与self一致，shape与self相同。
 * 支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND[（参考）](#参考)。
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND[（参考）](#参考)。
 * @param [in] negativeSlope: host侧的aclScalar，表示self<0时的斜率。
 * @param [in] selfIsResult: host侧的bool，表示self是否做为输出。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、DOUBLE。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND[（参考）](#参考)。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* negativeSlope, bool selfIsResult,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnLeakyReluBackward的第二段接口，用于执行计算。
 *
 * 算子功能：激活函数反向。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeakyReluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLeakyReluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEAKY_RELU_BACKWARD_H_