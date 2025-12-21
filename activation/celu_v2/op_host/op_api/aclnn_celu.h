/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_CELU_H_
#define OP_API_INC_LEVEL2_ACLNN_CELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对输入张量self中的每个元素x调用连续可微指数线性单元激活函数CELU，并将得到的结果存入输出张量out中。
 * 计算公式：如下
 * $$
 * CELU(x) = max(0, x) + min(0, \alpha \ast (exp(x / \alpha) - 1))
 * $$
 *
 * 实现说明：如下
 *
 * 计算图：如下
 *
 * ```mermaid
 * graph LR
 *     A[(Self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::CeluV2])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 *     G((alpha)) --> C
 * ```
 */

/**
 * @brief aclnnCelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] self: 表示CELU激活函数的输入，npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，支持非连续的
 * Tensor, 数据格式支持ND，数据维度不支持8维以上。
 * @param [in] alpha: 表示CELU激活函数的激活系数，host侧的aclScalar，数据类型需要是可转换为FLOAT的数据类型。
 * @param [in] out: 表示CELU激活函数的输出，npu
 * device侧的aclTensor，数据类型需要是self可转换的数据类型，shape需要与self一致，
 * 支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCeluGetWorkspaceSize(
    const aclTensor* self, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnCelu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCeluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceCelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] selfRef: 表示CELU激活函数的输入，npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16，支持非连续的Tensor，数据 格式支持ND，数据维度不支持8维以上。
 * @param [in] alpha: 表示CELU激活函数的激活系数，host侧的aclScalar，数据类型需要是可转换为FLOAT的数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceCeluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceCelu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceCeluGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceCelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_CELU_H_
