/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_
#define OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnEmbeddingDenseBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成Embedding的反向计算
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(grad)] -->B([l0op::Contiguous])
 *     B --> L([l0op::Cast])
 *     L --> C([l0op::EmbeddingDenseGrad])
 *     D[(indices)] -->E([l0op::Contiguous])
 *     E --> F([l0op::Cast])
 *     F --> C
 *     C --> M([l0op::Cast])
 *     M --> G([l0op::ViewCopy])
 *     G --> K[(out)]
 *     H((numWeights)) --> C
 *     I((paddingIdx)) --> C
 *     J((scaleGradByFreq)) --> C
 * ```
 *
 * @param [in] grad: npu
 * device侧的aclTensor，梯度Tensor，和正向的输出shape一致，npu
 * device侧的aclTensor，数据类型支持float、float16、bfloat16类型， 数据格式支持ND，比indices的维度多一。
 * @param [in] indices: npu
 * device侧的aclTensor，正向中需要映射到向量空间的索引张量，数据类型支持int32，支持非连续的Tensor，数据格式支持ND。
 * @param [in] numWeights: 向量空间的大小。
 * @param [in] paddingIdx:
 * 填充ID，默认为None，如果指定的话，将指定位置处的向量元素全部置为0，且paddingIdx对应的参数不会对梯度产生影响。
 * @param [in] scaleGradByFreq: 根据单词出现的频率，对梯度进行放缩，默认为False。
 * @param [out] out: npu
 * 反向输出Tensor，数据类型支持float32类型，数据格式仅支持2D。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(
    const aclTensor* grad, const aclTensor* indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnEmbeddingDenseBackward的第二段接口，用于执行计算。
 *
 * 算子功能：完成Embedding的反向计算
 *
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(grad)] -->B([l0op::Contiguous])
 *     B --> L([l0op::Cast])
 *     L --> C([l0op::EmbeddingDenseGrad])
 *     D[(indices)] -->E([l0op::Contiguous])
 *     E --> F([l0op::Cast])
 *     F --> C
 *     C --> M([l0op::Cast])
 *     M --> G([l0op::ViewCopy])
 *     G --> K[(out)]
 *     H((numWeights)) --> C
 *     I((paddingIdx)) --> C
 *     J((scaleGradByFreq)) --> C
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnEmbeddingDenseBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnEmbeddingDenseBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_
