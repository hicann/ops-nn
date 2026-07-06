/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_V2_EXPERIMENTAL_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_V2_EXPERIMENTAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the workspace size and executor for aclnnGatherV2.
 *
 * @param [in] self Source tensor to gather from.
 * @param [in] dim Dimension along which to gather.
 * @param [in] index Index tensor. Supported dtypes are int32 and int64.
 * @param [out] out Output tensor.
 * @param [out] workspaceSize Workspace size required by the second-stage interface.
 * @param [out] executor Executor used by the second-stage interface.
 * @return aclnnStatus ACL_SUCCESS on success, otherwise an error code.
 */
ACLNN_API aclnnStatus aclnnGatherV2GetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Executes aclnnGatherV2 with the workspace and executor returned by aclnnGatherV2GetWorkspaceSize.
 *
 * @param [in] workspace Device workspace pointer.
 * @param [in] workspaceSize Workspace size in bytes.
 * @param [in] executor Executor returned by aclnnGatherV2GetWorkspaceSize.
 * @param [in] stream Runtime stream.
 * @return aclnnStatus ACL_SUCCESS on success, otherwise an error code.
 */
ACLNN_API aclnnStatus aclnnGatherV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GATHER_V2_EXPERIMENTAL_H_
