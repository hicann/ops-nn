/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SELU_H_
#define OP_API_INC_LEVEL2_ACLNN_SELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成selu操作
 * @param [in] self: npu device侧的aclTensor, 数据类型支持INT8、INT32,
 * FLOAT、FLOAT16、shape为非空，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor, 数据类型支持INT8、INT32, FLOAT、FLOAT16, shape与self
 *  保持相同，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSeluGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief: aclnnSelu的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成selu操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSeluGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceSelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成selu操作
 * @param [in] selfRef: npu device侧的aclTensor, 数据类型支持INT8、INT32,
 * FLOAT、FLOAT16，shape为非空，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceSeluGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief: aclnnInplaceSelu的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor原地完成selu操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceSeluGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceSelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
