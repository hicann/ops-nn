/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_
#define OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFastGelu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：返回输入FastGelu的计算结果
 * 计算公式：
 * $$ FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} * \exp(0851 * (x_i - \left| x_i \right|)) $$
 *
 *
 * @param [in] self: 输入张量，npu device侧的aclTensor,
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] out: 输出张量，npu device侧的aclTensor,
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包括算子计算流程
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnFastGeluGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFastGelu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnFastGeluGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnFastGelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif /* OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_ */