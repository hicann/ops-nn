/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FOREACH_MINIMUM_SCALAR_LIST_H_
#define ACLNN_FOREACH_MINIMUM_SCALAR_LIST_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnForeachMinimumScalarList的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入张量列表按标量列表逐元素取最小值。
 * 计算公式：out_{i}=minimum(x_{i}, scalars_{i})
 * @domain aclnnop_math
 * @param [in]   x              输入Tensor列表，数据类型支持FLOAT，FLOAT16，INT32，BFLOAT16。数据格式支持ND。
 * @param [in]   scalars        输入标量列表，元素个数与`x`中Tensor个数一致。
 * @param [in]   out            输出Tensor列表，数据类型支持FLOAT，FLOAT16，INT32，BFLOAT16。数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor        返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus     返回状态码
 */
ACLNN_API aclnnStatus aclnnForeachMinimumScalarListGetWorkspaceSize(const aclTensorList* x,
                                                                    const aclScalarList* scalars,
                                                                    const aclTensorList* out, uint64_t* workspaceSize,
                                                                    aclOpExecutor** executor);
/**
 * @brief aclnnForeachMinimumScalarList的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnForeachMinimumScalarList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif // ACLNN_FOREACH_MINIMUM_SCALAR_LIST_H_
