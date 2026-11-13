/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SELU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_SELU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSeluBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：计算SELU反向梯度。
 * 数据类型支持FLOAT、FLOAT16、INT32、INT8、BFLOAT16，具体产品支持范围参见接口文档。
 * @param [in] gradOutput: npu device侧的上游梯度aclTensor，维度为1-8，支持空Tensor和非连续Tensor，
 * 数据格式支持ND。
 * @param [in] result: SELU正向输出，数据类型和shape与gradOutput相同，支持空Tensor和非连续Tensor，
 * 数据格式支持ND。
 * @param [out] gradInput: 反向梯度输出，数据类型和shape与gradOutput相同，支持空Tensor和非连续Tensor，
 * 数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSeluBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* result,
                                                        aclTensor* gradInput, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnSeluBackward的第二段接口，用于执行计算。
 *
 * 算子功能：计算SELU反向梯度。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnSeluBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSeluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
