/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_HARDSHRINK_BACKWARD_H_
#define OP_API_INC_HARDSHRINK_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardshrinkBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能： 对输入Tensor完成hardShrink backward操作
 * @param [in] gradOutput: 计算输入, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] self: 计算输入, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [in] lambd:计算输入, 指定hardShrinkBackward分段的阈值，数据类型支持FLOAT类型。
 * @param [in] gradInput:计算输出, 数据类型支持FLOAT、FLOAT16, 数据格式支持ND, 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码， 成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus aclnnHardshrinkBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnHardshrinkBackward的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成hardShrink backward操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnHardshrinkBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码,成功返回ACLNN_SUCCESS, 失败返回对应错误码。
 */
ACLNN_API aclnnStatus
aclnnHardshrinkBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_HARDSHRINK_BACKWARD_H_