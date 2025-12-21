/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELU_H_
#define OP_API_INC_LEVEL2_ACLNN_GELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：高斯误差线性单元激活函数
 * 计算公式：
 * $$ out_{i}=Gelu(self_{i})=self_{i}×Φ(self_{i}) $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Gelu])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: 待进行gelu计算的入参。npu device侧的aclTensor。
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和out一样，数据格式支持ND，shape必须和out一样，支持非连续的Tensor。
 * @param [in] out: gelu计算的出参。
 * npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和self一样，数据格式支持ND，shape必须和self一样，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeluGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnGelu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GELU_H_