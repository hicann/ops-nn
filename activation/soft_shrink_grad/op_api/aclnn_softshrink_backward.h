/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SOFT_SHRINK_BACKWARD_H_
#define OP_API_INC_SOFT_SHRINK_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSoftshrinkBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * @param [in] gradOutput: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * shape与self满足broadcast关系。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * shape与gradOutput满足broadcast关系。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [in] lambda: npu host侧的aclScalar，表示Softshrink计算的非负阈值。
 * @param [out] gradInput: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * shape与gradOutput和self的broadcast结果一致。支持[非连续的Tensor](#非连续Tensor说明)，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                              const aclScalar* lambda, aclTensor* gradInput,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSoftshrinkBackward的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnSoftshrinkBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSoftshrinkBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_SOFT_SHRINK_BACKWARD_H_
