/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_RELU_H_
#define OP_API_INC_RELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ output_i = xi if xi > 0 else 0
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Relu])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持
 * FLOAT、FLOAT16、INT8、INT32、UINT8、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持
 * FLOAT、FLOAT16、INT8、INT32、UINT8、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnReluGetWorkspaceSize(const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnRelu的第二段接口，用于执行计算。
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ output_i = xi if xi > 0 else 0
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Relu])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnReluGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief aclnnInplaceRelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ output_i = xi if xi > 0 else 0
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Relu])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
```
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持
FLOAT、FLOAT16、INT8、INT32、UINT8、BFLOAT16（在Ascend910及之前芯片上不支持该数据类型），
 * 支持非连续的Tensor，数据格式支持ND
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceReluGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRelu的第二段接口，用于执行计算。
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ output_i = xi if xi > 0 else 0
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Relu])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceReluGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RELU_H
