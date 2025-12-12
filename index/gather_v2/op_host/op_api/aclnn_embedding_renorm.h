/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_EMBEDDING_RENORM_H_
#define OP_API_INC_EMBEDDING_RENORM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEmbeddingRenorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：根据给定的max_norm和norm_type返回tensor在指定indices下的修正结果
 *
 * 计算公式：
 * 向量的范数计算公式如下，其中n为norm_type指定的范数值：
 * $$ ||X||_{p}=\sqrt[p]{\sum_{i=1}^nx_{i}^p} $$
 * $$ 其中X = (x_{1},x_{2}, ... , x_{n}) $$
 * 针对计算出的范数大于给定max_norm的场景，需要做归一化处理，对indices指定的0维元素乘以系数：
 * $$ scaler = \frac{max\_norm}{current\_norm + 1e^{-7}} $$
 *
 * 实现说明
 * api计算基本路径：
 *```mermaid
 * graph LR
 * A[(selfRef)]  --> B([l0op::contiguous])
 * B --> C([l0op::GatherV2])
 * D((axis=0)) --> C
 * E[(Indices)] --> F([l0op::contiguous])
 * F --> R([l0op::Reshape])
 * R --> C
 * C --> G([l0op::Renorm])
 * H((maxNorm)) --> G
 * I((normType)) --> G
 * J((dim=0)) --> G
 * U[(indicesGatherV2)] --> V([l0op::Contiguous])
 * V --> K([l0op::GatherV2])
 * L((axis=0)) --> K
 * G --> K
 * C --> M([l0op::Mul])
 * K --> M
 * B --> N([l0op::ScatterUpdate])
 * M --> N
 * R --> N
 * N --> O([l0op::ViewCopy])
 * O --> P[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT32类型。
 * 支持非连续的Tensor，数据格式支持ND，dimension必须为2。
 * @param [in] indices: npu
 * device侧的aclTensor，数据类型支持INT64。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] maxNorm: 数据类型支持double、float。
 * @param [in] normType: 数据类型支持double、float。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor* selfRef, const aclTensor* indices, double maxNorm,
                                                           double normType, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief aclnnEmbeddingRenorm的第二段接口，用于执行计算
 *
 * 算子功能：根据给定的max_norm和norm_type返回tensor在指定indices下的修正结果
 *
 * 计算公式：
 * 向量的范数计算公式如下，其中n为norm_type指定的范数值：
 * $$ ||X||_{p}=\sqrt[p]{\sum_{i=1}^nx_{i}^p} $$
 * $$ 其中X = (x_{1},x_{2}, ... , x_{n}) $$
 * 针对计算出的范数大于给定max_norm的场景，需要做归一化处理，对indices指定的0维元素乘以系数：
 * $$ scaler = \frac{max\_norm}{current\_norm + 1e^{-7}} $$
 *
 * 实现说明
 * api计算基本路径：
 *```mermaid
 * graph LR
 * A[(selfRef)]  --> B([l0op::contiguous])
 * B --> C([l0op::GatherV2])
 * D[(axis=0)] --> C
 * E[(Indices)] --> F([l0op::contiguous])
 * F --> C
 * C --> G([l0op::Renorm])
 * H((maxNorm)) --> G
 * I((normType)) --> G
 * J((dim=0)) --> G
 * F --> K([l0op::GatherV2])
 * L[(axis=0)] --> K
 * G --> K
 * C --> M([l0op::Mul])
 * K --> M
 * B --> N([l0op::ScatterUpdate])
 * M --> N
 * F --> N
 * N --> O([l0op::ViewCopy])
 * O --> P[(selfRef)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingRenorm(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
                                           const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EMBEDDING_RENORM_H_
