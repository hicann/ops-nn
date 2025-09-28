/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_INVERSE_H_
#define OP_API_INC_LEVEL2_ACLNN_INVERSE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：取方阵的逆矩阵。
 * 计算公式：如下
 * $$
 * {A}^{-1}A = A{A}^{-1} = {I}_{n}
 * $$
 * 其中A是输入张量，${A}^{-1}$是A的逆，${I}_{n}$是n维的单位矩阵。
 *
 * 计算图一：如下
 * 场景：输入self的数据类型不是FLOAT16。
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::MatrixInverse])
 *   C --> D([l0op::Cast])
 *   D --> E([l0op::ViewCopy])
 *   E --> F[(out)]
 * ```
 *
 * 计算图二：如下
 * 场景：当输入self的数据类型是FLOAT16时，需要将FLOAT16转成FLOAT32，传给算子计算。
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Cast])
 *   C --> D([l0op::MatrixInverse])
 *   D --> E([l0op::Cast])
 *   E --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 */

/**
 * @brief aclnnInverse的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128、FLOAT16，shape至少是2维，
 * 且最后两维的大小必须相同，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128，且数据
 * 类型需要是self可转换的数据类型，shape需要与self的shape一致，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnInverse的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInverseGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInverse(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_INVERSE_H_