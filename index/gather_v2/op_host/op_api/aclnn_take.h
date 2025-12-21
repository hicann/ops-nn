/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_TAKE_H_
#define OP_API_INC_LEVEL2_ACLNN_TAKE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTake的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：将输入的self张量视为一维数组，把index的值当作索引，从self中取值，输出shape与index一致的Tensor。
 * 计算公式：其中下标i表示从0遍历到index元素个数-1
 * $$ out_{i} = self_{index[i]}  $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous]) --> D([l0op::Gather])
 * I[(index)] --> IC([l0op::Contiguous]) --> D --> F1([l0op::ViewCopy]) --> J[(out)]
 * ```
 *
 * @param [in] self: 待进行take计算的入参。npu device侧的aclTensor,
 *     数据类型支持UINT64、INT64、UINT32、INT32、FLOAT32、UINT16、INT16、FLOAT16、INT8、UINT8、DOUBLE、COMPLEX64、COMPLEX128、BOOL，
 *     数据格式支持ND，支持非连续的Tensor，支持维度高于8的场景。
 * @param [in] index: take计算的入参。npu device侧的aclTensor，数据类型支持INT32、INT64。数据格式为ND。
 * @param [in] out: take计算的出参。npu device侧的aclTensor，数据类型同self，shape与index一致，数据格式为ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor* self, const aclTensor* index, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnTake的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTakeGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTake(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TAKE_H_
