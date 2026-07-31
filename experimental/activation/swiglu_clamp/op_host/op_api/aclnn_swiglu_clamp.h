/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_SWIGLU_CLAMP_H_
#define OP_API_INC_ACLNN_SWIGLU_CLAMP_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSwigluClamp phase 1: compute workspace size.
 * @domain aclnn_ops_infer
 *
 * out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit), where gate = x[..., :N],
 * up = x[..., N:], out shape = x shape with the last dim halved.
 *
 * @param [in]  x: npu aclTensor, dtype supports FLOAT16/BFLOAT16/FLOAT, format ND.
 * @param [in]  limit: clamp threshold, must be > 0 (Step-3.7 uses 7.0).
 * @param [out] y: npu aclTensor, same dtype as x, shape = x shape with last dim halved.
 * @param [out] workspaceSize: workspace size to allocate on the device side.
 * @param [out] executor: op executor holding the compute flow.
 * @return aclnnStatus.
 */
ACLNN_API aclnnStatus aclnnSwigluClampGetWorkspaceSize(const aclTensor* x, double limit, aclTensor* y,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSwigluClamp phase 2: execute.
 *
 * @param [in] workspace: device workspace base address.
 * @param [in] workspaceSize: device workspace size, from aclnnSwigluClampGetWorkspaceSize.
 * @param [in] executor: op executor from phase 1.
 * @param [in] stream: acl stream.
 * @return aclnnStatus.
 */
ACLNN_API aclnnStatus aclnnSwigluClamp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_SWIGLU_CLAMP_H_
