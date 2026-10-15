/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_FOREACH_ADDCMUL_SCALAR_H_
#define OP_API_INC_ACLNN_FOREACH_ADDCMUL_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnForeachAddcmulScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入张量列表的每个张量计算x1与x2*x3乘以标量值scalar的和后输出。
 * 计算公式：out_{i}=x1_{i}+scalar*(x2_{i}*x3_{i})
 * @domain aclnnop_math
 * @param [in]   x1              输入Tensor，数据类型支持FLOAT，FLOAT16，INT32，BFLOAT16。数据格式支持ND。
 * @param [in]   x2              输入Tensor，数据类型与格式与入参`x1`一致。数据格式支持ND。
 * @param [in]   x3              输入Tensor，数据类型与格式与入参`x1`一致。数据格式支持ND。
 * @param [in]   scalar          输入Scalar Tensor，数据类型与入参`x1`的数据类型具有对应关系（BFLOAT16映射FLOAT32，
 *                               整型映射INT32），元素个数为1。数据格式支持ND。
 * @param [in]   out             输出Tensor，数据类型和格式与入参`x1`一致。数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor        返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus     返回状态码
 */
ACLNN_API aclnnStatus aclnnForeachAddcmulScalarGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                                const aclTensorList* x3, const aclTensor* scalar,
                                                                aclTensorList* out, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor);

/**
 * @brief aclnnForeachAddcmulScalar的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnForeachAddcmulScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_FOREACH_ADDCMUL_SCALAR_H_
