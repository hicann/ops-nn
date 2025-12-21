/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HEAVISIDE_H_
#define OP_API_INC_HEAVISIDE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHeavisideGetWorkspaceSize的第一段接口,根据具体的计算流程,计算workspace大小。
 * @domain aclnnop_ops_infer
 * 功能描述：计算输入input中每个元素的Heaviside阶跃函数。
 * 计算公式：如下所示
 * $$
 * \text{heaviside}(\text{input}, \text{values}) =
 * \begin{cases}
 * 0, & \text{如果 input} < 0 \\
 * \text{values}, & \text{如果 input} = 0 \\
 * 1, & \text{如果 input} > 0
 * \end{cases}
 * $$
 *
 * @param [in] input:  npu
 * device侧的aclTensor，数据类型支持FLOAT，FLOAT16，BFLOAT16，shape需要与values满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与values一致。
 * @param [in] values: npu
 * device侧的aclTensor，数据类型支持FLOAT，FLOAT16，BFLOAT16，shape需要与input满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与input一致。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT，FLOAT16，BFLOAT16，shape需要是input与values
 * broadcast之后的shape，数据格式支持ND，且数据格式需要与input一致。支持非连续Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnHeavisideGetWorkspaceSize(
    const aclTensor* input, const aclTensor* values, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHeaviside的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnForeachAbsGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnHeaviside(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_HEAVISIDE_H_