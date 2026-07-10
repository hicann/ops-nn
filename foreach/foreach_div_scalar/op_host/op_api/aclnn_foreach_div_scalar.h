/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_FOREACH_DIV_SCALAR_H_
#define OP_API_INC_ACLNN_FOREACH_DIV_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnForeachDivScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入张量列表的每个张量除以标量值scalar后输出。
 * 计算公式：out_{i}=x_{i}/scalar
 * @domain aclnnop_math
 * @param [in]   x               输入Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。数据格式支持ND。
 * @param [in]   scalar          输入Scalar Tensor，数据类型支持FLOAT，FLOAT16。数据格式支持ND。
 * @param [in]   out             输出Tensor，数据类型支持FLOAT，FLOAT16，BFLOAT16。数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor        返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus     返回状态码
 */
ACLNN_API aclnnStatus aclnnForeachDivScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar,
                                                            aclTensorList* out, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

/**
 * @brief aclnnForeachDivScalar的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnForeachDivScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_FOREACH_DIV_SCALAR_H_
