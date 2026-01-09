/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_EMBEDDING_H_
#define OP_API_INC_EMBEDDING_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEmbedding的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：把数据集合映射到向量空间，进而将数据进行量化
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(weight)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GatherV2])
 *     D[(indices)] --> E([l0op::Contiguous]) --> C
 *     F((dim=0)) --> G[(dimTensor)] --> C
 *     C --> H([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] weight: npu
 * device侧的aclTensor，数据类型支持BFLOAT16,
 * FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE、COMPLEX64、
 * COMPLEX128，支持非连续Tensor，数据格式支持ND。
 * @param [in] indices: npu
 * device侧的aclTensor，数据类型支持INT64、INT32，支持非连续Tensor，数据格式支持ND。
 * @param [out] out: npu
 * device侧的aclTensor，数据类型同weight，数据格式支持ND，比indices的维度多1。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor* weight, const aclTensor* indices,
                                                     const aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnEmbedding的第二段接口，用于执行计算。
 *
 * * 算子功能：把数据集合映射到向量空间，进而将数据进行量化
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(weight)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GatherV2])
 *     D[(indices)] --> E([l0op::Contiguous]) --> C
 *     F((dim=0)) --> G[(dimTensor)] --> C
 *     C --> H([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnEmbeddingGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbedding(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EMBEDDING_H_