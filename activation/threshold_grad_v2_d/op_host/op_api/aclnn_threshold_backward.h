/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_THRESHOLD_BACKWARG_H_
#define OP_API_INC_THRESHOLD_BACKWARG_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnThresholdBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成threshold_forward对应反向
 * 计算公式：
 * res(i) = gradOutput(i) if self(i) > threshold else 0
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::ThresholdGradV2D or l0op::ReluGrad])
 * D[(grad_output)] -->  E([l0op::Contiguous])
 * E  --> C
 * F[(threshold)] --> C
 * C--> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 *
 * @param [in] gradOutput: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8，shape需要与self一致。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] threshold: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT8、UINT8，shape需要与self一致。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                             const aclScalar *threshold, aclTensor *out,
                                                             uint64_t *workspaceSize, aclOpExecutor **executor);
/**
 * @brief aclnnAdd的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnThresholdBackwardGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnThresholdBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_THRESHOLD_BACKWARG_H_
