/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_HARDSHRINK_H_
#define OP_API_INC_HARDSHRINK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardshrink的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 功能描述：以元素为单位，强制收缩λ范围内的元素。
 * 计算公式：如下
 * $$
 * Hardshrink(x)=
 * \begin{cases}
 * x, if x > λ \\
 * x, if x < -λ \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [in]   lambd
 * 输入Scalar，数据类型支持FLOAT。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnHardshrinkGetWorkspaceSize(
    const aclTensor* self, const aclScalar* lambd, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnHardshrink的第二段接口，用于执行计算。
 * 功能描述：以元素为单位，强制收缩λ范围内的元素。
 * 计算公式：如下
 * $$
 * Hardshrink(x)=
 * \begin{cases}
 * x, if x > λ \\
 * x, if x < -λ \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * 实现说明：
 * api计算的基本路径：
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B -->C([l0op::HardShrink])
    C -->D([l0op::Cast])
    D -->E([l0op::ViewCopy])
    E -->F[(Out)]

    G((lambd)) -->C
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnHardshrinkGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnHardshrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_HARDSHRINK_H_
