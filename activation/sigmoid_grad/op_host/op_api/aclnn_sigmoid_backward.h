/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SIGMOID_BACKWARD_H_
#define OP_API_INC_SIGMOID_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSigmoidgBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能： 对输入Tensor完成sigmoid backward操作
 * @param [in] gradOutput: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128,
 * shape为非空， 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] output: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128,
 * shape与gradOutput保持相同，数据格式支持ND, 支持非连续的Tensor。
 * @param [in] gradInput: npu device侧的aclTensor, 数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128，
 * shape与gradOutput保持相同，数据格式支持ND, 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* output,
                                                           aclTensor* gradInput, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief: aclnnSigmoidBackward的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成sigmoid backward操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSigmoidBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSigmoidBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SIGMOID_BACKWARD_H_