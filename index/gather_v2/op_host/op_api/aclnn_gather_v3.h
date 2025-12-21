/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：从输入Tensor的指定维度0，按index中的下标序号提取元素，保存到out Tensor中。
 * 计算公式：
 * 以输入为三维张量为例：
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(0) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy]) --> Out[(out)]
 * ```
 */

/**
 * 算子功能：从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。
 * 计算公式：
 * 当batchDims 为0，以输入为三维张量为例：
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * dim为0：   I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$
 * dim为1：   J=index[j];  &nbsp;&nbsp;&nbsp;    y$[l][j][n]$ = x$[l][J][n]$
 * dim为2：   K=index[k]; &nbsp;  y$[l][m][k]$ = x$[l][m][K]$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(dim) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy]) --> Out[(out)]
 * ```
 * batchDims 不为0：
 * batchDims 为n, dim 为m(m >= n),batchDims 之前的shape大小为k，相当与进行k次batchDims=0，
 * dim=m-n的gather操作。
 */

/**
 * @brief aclnnGatherV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE、
 * COMPLEX64、COMPLEX128、BFLOAT16，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] dim: host侧的整数，数据类型支持INT64。
 * @param [in] index: npu
 * device侧的aclTensor，数据类型支持INT64、INT32，支持非连续的Tensor，数据格式支持ND，数据维度不支持 8维以上。
 * @param [in] batchDims: host侧的整数，数据类型支持INT64。
 * @param [in] mode: host侧的整数，数据类型支持INT64。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE、
 * COMPLEX64、COMPLEX128、BFLOAT16，数据类型需要与self一致，维数等于self维数与index维数之和减一，除dim维扩展为跟index的shape
 * 一样外，其他维长度与self相应维一致，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index, int64_t batchDims,
                                                    int64_t mode, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnGatherV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGatherV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGatherV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_
