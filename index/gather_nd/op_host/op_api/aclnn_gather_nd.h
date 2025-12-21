/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对于维度为**r≥1**的输入张量`self`，和维度**q≥1**的输入张量`indices`，将数据切片收集到维度为**(q-1) + (r -
 * indices_shape[-1])**
 * 的输出张量out中。indices是一个**q**维的整型张量，可视作一个**q-1**维的由**索引对**构成的特殊张量(每个**索引对**是一个长度为**indices_shape[-1]**
 * 的一维张量，每个**索引对**指向**self**中一个切片)，具体计算逻辑为：
 *  1) 如果**indices_shape[-1]** > **r**，不合法场景。
 *  2) 如果**indices_shape[-1]** = **r**，则输出张量out的维度为**q-1**，即out的shape为 **[indices_shape[0:q-2]]**，
 *     out中元素为self的**索引对**位置的元素。（见例1）
 *  3) 如果**indices_shape[-1]** < **r**，则输出张量out的维度为 **(q-1) + (r -
 * indices_shape[-1])**，设**c**=**r**-**indices_shape[-1]**， 即out的shape为
 * **[indices_shape[0:q-2],self_shape[r-c:r-1]]**，`out`由`self`的**索引对**位置的切片组成。（见例2,3,4）
 * 关于**r**、**q**、**indices_shape[-1]** 的一些限制条件如下：
 *  1) 必须满足**r**≥1，**q**≥1。
 *  2) **indices_shape[-1]**的值必须满足在1(包含)和**r**(包含)之间。
 *  3) `indices`的每个元素，必须在[-**s**, **s-1**]范围内(**s**为**self_shape**各个轴上的值)，
 *     即-**self_shape[i]**≤indices[...,i]≤**self_shape[i]**-1。
 * 示例
 *  例1：
 *   self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
 *   indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
 *   out: [0, 3]                 # out_shape=[2]
 *  例2：
 *   self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
 *   indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
 *   out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
 *  例3：
 *   self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
 *   indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
 *   out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
 *  例4：
 *   self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
 *   indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
 *   out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   C[(indices)] --> D([l0op::Contiguous])
 *   B --> E([l0op::GatherNd])
 *   D --> E
 *   E --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 */

/**
 * @brief aclnnGatherNd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持INT64、INT32、INT8、UINT8、BOOL、FLOAT、FLOAT16、BFLOAT16，支
 * 持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] index: npu
 * device侧的aclTensor，数据类型支持INT64、INT32，支持非连续的Tensor，数据格式支持ND，数据维度不支持 8维以上。
 * @param [in] negativeIndexSupport: 属性，表示是否支持负数场景，数据类型支持bool。
 * @param [in] out: npu device侧的aclTensor，数据类型支持INT64、INT32、INT8、UINT8、BOOL、FLOAT、FLOAT16、BFLOAT16，数据
 * 类型需要与self一致，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGatherNdGetWorkspaceSize(const aclTensor* self, const aclTensor* indices,
                                                    bool negativeIndexSupport, aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief aclnnGatherNd的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGatherNdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGatherNd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_
